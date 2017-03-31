function onehot!(df, varname)
    for keyword in unique(df[:, varname])[1:end-1]
        sym_keyword = Symbol(keyword)
        df[sym_keyword] = zeros(Int, size(df, 1))
        for i in 1:size(df, 1)
            if df[i, varname] == keyword
                df[i, sym_keyword] = 1
            end
        end
    end
    delete!(df, varname)
end


function binarize!(df, varname)
    tmp = zeros(Int, size(df, 1))
    for (i, keyword) in enumerate(unique(df[varname]))
        flt = df[varname] .== keyword
        tmp[flt] = i-1
        df[varname] = tmp
    end
end


function normalize!(df, varnames::Array)
    for varname in varnames
		normalize!(df, varname)
    end
end


function normalize!(df, varname::Symbol)
    mu = mean(df[varname])
    s = std(df[varname])
    df[varname] = (df[varname] .- mu) ./ s
end


function min_max_norm!(df, varnames::Array)
	for varname in varnames
	    min_max_norm!(df, varname)
	end
end


function min_max_norm!(df, varname::Symbol)
    ymin, ymax = extrema(df[varname])
    dy = ymax - ymin
    df[varname] = (df[varname] .- ymin) ./ dy
end


function train_test_split(X, y, at=0.8)
    nTrain = round(Int, size(X, 1) * 0.8)
    nTest = size(X, 1) - nTrain
    
    idx = randperm(size(X, 1))
    itrain = idx[1:nTrain]
    itest = idx[nTrain+1:end]
    Xtrain = X[itrain, :]
    ytrain = y[itrain]
    Xtest = X[itest, :]
    ytest = y[itest]
    
    return Xtrain, ytrain, Xtest, ytest
end
    
    
