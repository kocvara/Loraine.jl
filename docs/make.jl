# Inspired from https://github.com/jump-dev/JuMP.jl/blob/master/docs/make.jl

using Loraine
using Documenter

makedocs(
    sitename = "Loraine",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    strict = true,
    pages = [
        "Index" => "index.md",
        "Overview" => "overview.md",
        "Loraine Options" => "Loraine_options.md",
        "Low-rank Solutions" => "low-rank_solutions.md",
    ],
    # The following ensures that we only include the docstrings from
    # this module for functions define in Base that we overwrite.
    modules = [Loraine]
)

deploydocs(
    repo   = "github.com/kocvara/Loraine.jl.git",
    target = "build",
    # push_preview = true,
)