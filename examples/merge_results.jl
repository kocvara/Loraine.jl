# Merge two benchmark CSVs (e.g. one from `main`, one from `nlpmodel`) produced
# by `benchmark_sdplib.jl`, and print a GitHub-ready markdown comparison table.
#
# Needs CSV.jl and DataFrames.jl. Run in an environment that has them, e.g.:
#
#     julia --project=@bench examples/merge_results.jl results_main.csv results_nlpmodel.csv
#
# (where `@bench` is any env with `CSV` and `DataFrames` added).
#
# Args: [1] baseline CSV (e.g. main), [2] comparison CSV (e.g. nlpmodel).
# The column labels are taken from each CSV's `label` field.

using CSV
using DataFrames
import Printf

function load(path)
    df = CSV.read(path, DataFrame)
    return df, isempty(df.label) ? basename(path) : first(df.label)
end

function fmt_time(t)
    (ismissing(t) || !isfinite(t)) && return "—"
    return Printf.@sprintf("%.3f", t)
end

function fmt_speedup(s)
    (ismissing(s) || !isfinite(s)) && return "—"
    return Printf.@sprintf("%.2f×", s)
end

function fmt_obj(o)
    (ismissing(o) || !isfinite(o)) && return "—"
    return Printf.@sprintf("%.6g", o)
end

function main(baseline = ARGS[1], comparison = ARGS[2])
    base, base_label = load(baseline)
    comp, comp_label = load(comparison)

    # Compare on `solve_time_s` (the solver's own time, free of compilation and
    # the worker harness) rather than the wall-clock `time_s`.
    a = select(
        base,
        :problem,
        :n,
        :m,
        :solve_time_s => :t_base,
        :objective => :obj_base,
        :status => :st_base,
    )
    b = select(
        comp,
        :problem,
        :solve_time_s => :t_comp,
        :objective => :obj_comp,
        :status => :st_comp,
    )
    df = outerjoin(a, b, on = :problem)
    df.speedup = df.t_base ./ df.t_comp
    sort!(df, [:speedup])

    # --- markdown table ---
    header = "| problem | n | m | $base_label (s) | $comp_label (s) | speedup | obj $base_label | obj $comp_label |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    println(header)
    println(sep)
    for r in eachrow(df)
        println(
            "| ",
            r.problem,
            " | ",
            ismissing(r.n) ? "—" : r.n,
            " | ",
            ismissing(r.m) ? "—" : r.m,
            " | ",
            fmt_time(r.t_base),
            " | ",
            fmt_time(r.t_comp),
            " | ",
            fmt_speedup(r.speedup),
            " | ",
            fmt_obj(r.obj_base),
            " | ",
            fmt_obj(r.obj_comp),
            " |",
        )
    end

    # --- summary over problems with a real solve time on both sides ---
    # (drops `missing` from the join and `NaN` from TIMEOUT/ERROR rows)
    isfin(x) = !ismissing(x) && isfinite(x)
    both = filter(row -> isfin(row.t_base) && isfin(row.t_comp), df)
    tot_base = sum(both.t_base)
    tot_comp = sum(both.t_comp)
    geo = isempty(both.speedup) ? NaN :
          exp(sum(log, both.speedup) / nrow(both))
    println()
    println(
        "**",
        nrow(both),
        " problems solved by both** — total time: ",
        fmt_time(tot_base),
        "s ($base_label) vs ",
        fmt_time(tot_comp),
        "s ($comp_label); overall ",
        fmt_speedup(tot_base / tot_comp),
        ", geometric-mean per-problem ",
        fmt_speedup(geo),
        ".",
    )
end

#main()
