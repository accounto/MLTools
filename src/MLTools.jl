__precompile__()

module MLTools

include("mls_datautils.jl")
include("mls_algorithms.jl")

export onehot!,
       binarize!,
       normalize!,
       min_max_norm!,
       
       train_test_split,

       train!,
       fit!,
       predict,
       predict_proba,

       LogisticRegression
end
