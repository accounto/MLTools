using Optim

immutable LogisticRegression
  X::Matrix{Float64}
  y::Vector{Float64}
  theta::Vector{Float64}
end


function predict_proba(X::Matrix, model::LogisticRegression)
  s = X * model.theta
  h = logit.(s)
end


function predict(X::Matrix, model::LogisticRegression)
  h = predict_proba(X, model)
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
      h = predict_proba(model.X, model) 
      score = sum(-model.y .* log.(h) .- (1 .- model.y) .* log.(1 .- h))
  end
  x0 = zeros(model.theta)
  res = optimize(cost, x0, solver)
  model.theta[:] = res.minimizer
  return res
end

fit!(model::LogisticRegression) = fit!(model, LBFGS())

train!(model::LogisticRegression) = fit!(model::LogisticRegression)

