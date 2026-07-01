# Benchmark Loraine on the SDPLIB test set and write a CSV of results.
#
# Usage (run once per branch, from the Loraine repo root):
#
#     julia --project=. examples/benchmark_sdplib.jl results_main.csv main
#     julia --project=. examples/benchmark_sdplib.jl results_nlpmodel.csv nlpmodel
#
# Args:  [1] output CSV path   (default "sdplib_results.csv")
#        [2] label for this run (default "loraine")  -> stored in the CSV
#
# Knobs (environment variables):
#   SDPLIB_MAXN     max matrix size  n to include (default 1000)
#   SDPLIB_MAXM     max #constraints m to include (default 3000)
#   SDPLIB_KIT      Loraine `kit` (0 = direct, 1 = CG; default 0)
#   SDPLIB_TIMEOUT  per-problem wall-clock limit in seconds (default 60)
#   SDPLIB_ONLY     comma-separated list of problem names to restrict to
#
# Each solve runs on a worker process so that a problem exceeding the timeout
# (or one that segfaults, e.g. the `kit=1` path) is force-killed without taking
# the whole benchmark down; the worker is then respawned for the next problem.
#
# Only options common to both `main` and `nlpmodel` are used, so the same
# script produces comparable CSVs on either branch. The CSV is written with a
# plain `print` (no CSV.jl dependency), and it needs only `JuMP` and `Loraine`.

using Distributed
import Printf

const DATA = joinpath(@__DIR__, "SDPLIB", "data")
const README = joinpath(@__DIR__, "SDPLIB", "README.md")

# Parse the README table `| name | m | n | optimal | notes |` into
# name => (m, n, optimal). The only non-numeric optima are the four `inf*`
# (in)feasibility problems; anything else unexpected is a hard error.
function read_reference()
    isfile(README) || error("SDPLIB README not found at $README")
    ref = Dict{String,NamedTuple{(:m, :n, :opt),Tuple{Int,Int,Float64}}}()
    for line in eachline(README)
        startswith(line, "|") || continue # non-table line
        cells = strip.(split(line, "|")) # `| a | b |` -> ["", "a", "b", ""]
        name = cells[2]
        # Skip the header and the `| --- |` separator, parse everything else.
        (name == "Problem" || startswith(name, "-")) && continue
        length(cells) == 7 || error("Unexpected SDPLIB README row: $line")
        # README has a typo "eqaulG11" for the file "equalG11".
        name = name == "eqaulG11" ? "equalG11" : name
        opt = if cells[5] in ("dual infeasible", "primal infeasible")
            NaN
        else
            parse(Float64, cells[5])
        end
        ref[name] =
            (m = parse(Int, cells[3]), n = parse(Int, cells[4]), opt = opt)
    end
    return ref
end

# Spawn a worker that shares our project and knows how to solve one problem.
# Return both the Distributed id and the OS pid (so we can SIGKILL a stuck one).
function start_worker()
    pid = addprocs(1; exeflags = `--project=$(Base.active_project())`)[1]
    ospid = remotecall_fetch(getpid, pid)
    remotecall_wait(Core.eval, pid, Main, quote
        using JuMP
        import Loraine
        function solve_problem(path, kit)
            model = read_from_file(path)
            set_optimizer(model, Loraine.Optimizer{Float64})
            set_attribute(model, "kit", kit)
            set_attribute(model, "eDIMACS", 1e-6)
            set_attribute(model, "verb", 0)
            set_attribute(model, "maxit", 100)
            set_attribute(model, "datasparsity", 8)
            optimize!(model) # warm up: absorb this problem's compilation
            wall = @elapsed optimize!(model)
            return (
                wall,
                solve_time(model), # solver's own `SolveTimeSec`
                barrier_iterations(model),
                objective_value(model),
                string(termination_status(model)),
            )
        end
    end)
    return pid, ospid
end

launch(pid, path, kit) =
    remotecall((p, k) -> Main.solve_problem(p, k), pid, path, kit)

# Solve with a wall-clock cap. `isready` on a *remote* Future blocks while the
# worker is busy (it has to query the worker), so we instead let an async task
# `fetch` into a *local* Channel and poll that with `timedwait`.
# Returns (:ok, (t, obj, status)) | (:error, exception) | (:timeout, nothing).
function solve_capped(pid, path, kit, timeout)
    ch = Channel{Any}(1)
    @async put!(ch, try
        fetch(launch(pid, path, kit))
    catch err
        err
    end)
    if timedwait(() -> isready(ch), timeout; pollint = 0.2) === :ok
        r = take!(ch)
        return r isa Exception ? (:error, r) : (:ok, r)
    end
    return (:timeout, nothing)
end

function bench(label;
    out = label * ".csv",
    maxn = parse(Int, get(ENV, "SDPLIB_MAXN", "1000")),
    maxm = parse(Int, get(ENV, "SDPLIB_MAXM", "3000")),
    kit = parse(Int, get(ENV, "SDPLIB_KIT", "0")),
    timeout = parse(Float64, get(ENV, "SDPLIB_TIMEOUT", "60")),
)
    only = haskey(ENV, "SDPLIB_ONLY") ? Set(split(ENV["SDPLIB_ONLY"], ",")) :
           nothing
    ref = read_reference()

    problems = String[]
    for f in sort(readdir(DATA))
        endswith(f, ".dat-s") || continue
        name = f[1:(end-length(".dat-s"))]
        isnothing(only) || name in only || continue
        r = ref[name] # every data file is in the README
        (r.n <= maxn && r.m <= maxm) || continue
        push!(problems, name)
    end
    # smallest first, so quick problems report early
    sort!(problems, by = p -> ref[p].n)
    isempty(problems) && return

    pid, ospid = start_worker()
    # Warm up compilation on the smallest problem so timings exclude it.
    warmup = joinpath(DATA, problems[1] * ".dat-s")
    fetch(launch(pid, warmup, kit))

    # Append so re-running accumulates more rows (one solve = one row); the
    # minimum over rows is taken later, in `merge_results.jl`.
    write_header = !isfile(out) || filesize(out) == 0
    open(out, "a") do io
        if write_header
            println(
                io,
                "label,problem,m,n,status,time_s,solve_time_s,iterations,objective,optimal,rel_gap",
            )
        end
        ntot = length(problems)
        for (idx, name) in enumerate(problems)
            r = ref[name]
            status, t, st, iters, obj = "ok", NaN, NaN, -1, NaN
            outcome, payload =
                solve_capped(pid, joinpath(DATA, name * ".dat-s"), kit, timeout)
            if outcome === :ok
                t, st, iters, obj, status = payload
            elseif outcome === :error
                status = "ERROR: " * sprint(showerror, payload)[1:min(end, 60)]
            else
                # Stuck: SIGKILL the worker's OS process, then respawn a fresh
                # one (and re-warm it) for the remaining problems.
                run(`kill -9 $ospid`)
                rmprocs(pid; waitfor = 0)
                status, t = "TIMEOUT", timeout
                pid, ospid = start_worker()
                fetch(launch(pid, warmup, kit))
            end
            gap = isfinite(obj) && isfinite(r.opt) ?
                  abs(obj - r.opt) / max(1, abs(r.opt)) : NaN
            println(
                io,
                join(
                    [
                        label,
                        name,
                        r.m,
                        r.n,
                        "\"$status\"",
                        round(t, sigdigits = 5),
                        round(st, sigdigits = 5),
                        iters,
                        obj,
                        r.opt,
                        gap,
                    ],
                    ",",
                ),
            )
            Printf.@printf(
                "[%2d/%2d] %-12s n=%-5d m=%-5d  %8.3fs (solve %8.3fs, %3d it)  obj=%-14.6g %s\n",
                idx,
                ntot,
                name,
                r.n,
                r.m,
                t,
                st,
                iters,
                obj,
                status,
            )
            flush(io)
        end
    end
    rmprocs(pid)
    println("\nWrote ", out)
end
