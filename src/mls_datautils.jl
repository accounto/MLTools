function onehot!(df, varname)
    for keyword in unique(df[:, varname])
        sym_keyword = Symbol(keyword)
        df[sym_keyword] = zeros(Int, size(df, 1))
        for i in 1:size(df, 1)
            if df[i, varname] == keyword
                df[i, sym_keyword] = 1
            end
        end
    end
end

function binarize!(df, varname)
    tmp = zeros(Int, size(df, 1))
    for (i, keyword) in enumerate(unique(df[varname]))
        flt = df[varname] .== keyword
        tmp[flt] = i-1
    end
    df[varname] = tmp
end
