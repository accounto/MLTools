__precompile__()
module MLTools

include("mls_datautils.jl")
include("mls_algorithms.jl")

export onehot!,
       binarize!,

       train!,
       fit!,
       predict,
       predict_proba,

       LogisticRegression
end
