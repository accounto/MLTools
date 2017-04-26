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
    mu_vec = Float64[]
    s_vec = Float64[]
    for varname in varnames
	mu, s = normalize!(df, varname)
        push!(mu_vec, mu)
        push!(s_vec, s)
    end
    mu_vec, s_vec
end


function normalize!(df, varname::Symbol)
    mu = mean(df[varname])
    s = std(df[varname])
    normalize!(df, varname, mu, s)
    return mu, s
end


function normalize!(df, varname, mu, s)
    df[varname] = (df[varname] .- mu) ./ s
end


function normalize!(df, varnames::Array, mu_vec, s_vec)
    for (ivar, varname) in enumerate(varnames)
        mu, s = normalize!(df, varname, mu_vec[ivar], s_vec[ivar])
    end
    mu_vec, s_vec
end

function min_max_norm!(df, varnames::Array)
    lower = Float64[]
    upper = Float64[]
    for varname in varnames
        ymin, ymax = extrema(df[varname])
        dy = ymax - ymin
        min_max_norm!(df, varname, ymin, ymax)
        push!(lower, ymin)
        push!(upper, ymax)
    end
    lower, upper
end


function min_max_norm!(df, varname::Symbol, ymin, ymax)
    df[varname] = (df[varname] .- ymin) ./ dy
    ymin, ymax
end


function min_max_norm!(df, varnames::Array, lower, upper)
    for (ivar, varname) in enumerate(varnames)
        ymin, ymax = min_max_norm!(df, varname, lower[ivar], upper[ivar])
        push!(lower, ymin)
        push!(upper, ymax)
    end
    lower, upper
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
    
    
