using Test

function run_examples(dir)
    for file in readdir(dir)
        if endswith(file, ".jl") && !(file in ["benchmark_sdplib.jl", "merge_results.jl"])
            @testset "$file" begin
                include(joinpath(dir, file))
            end
        end
    end
end
run_examples(joinpath(dirname(@__DIR__), "examples"))

include("loraine.jl")
include("MOI_wrapper.jl")
