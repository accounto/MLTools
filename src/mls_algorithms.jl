using Optim

immutable LogisticRegression
  X::Matrix{Float64}
  y::Vector{Float64}
  theta::Vector{Float64}
end

LogisticRegression(X, y) = LogisticRegression(X, y, rand(size(X, 2)))


function predict_proba(model::LogisticRegression, X::Matrix)
  s = X * model.theta
  h = logit.(s)
end


function predict(model::LogisticRegression, X::Matrix)
  h = predict_proba(model, X)
  h[h .>= 0.5] = 1.0
  h[h .< 0.5] = 0.0
  return h
end


function logit(s)
    return 1 / (1 + exp(-s))
end


function fit!(model::LogisticRegression, solver)
  function cost(theta)
      model.theta[:] = theta
      s = model.X * model.theta
      h = predict_proba(model, model.X)
      score = sum(-model.y .* log.(h) .- (1 .- model.y) .* log.(1 .- h))
  end
  x0 = zeros(model.theta)
  res = optimize(cost, x0, solver)
  model.theta[:] = res.minimizer
  return res
end

train!(model::LogisticRegression, solver) = fit!(model::LogisticRegression, solver)
