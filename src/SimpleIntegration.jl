module SimpleIntegration

using Base: Fix1, Fix2

using OhMyThreads

export integrate, norm_L1, norm_L2

function integrate(f, knots_or_dx; threaded=true)
  if !threaded
    return integrate_1d_serial_impl(f, knots_or_dx)
  else
    return integrate_1d_threaded_impl(f, knots_or_dx)
  end
end

function integrate(f, knots_or_dx, knots_or_dy; threaded=true)
  if !threaded
    return integrate_2d_serial_impl(f, knots_or_dx, knots_or_dy)
  else
    return integrate_2d_threaded_impl(f, knots_or_dx, knots_or_dy)
  end
end

norm_L1(f::Function, knots...; threaded=true) = integrate((arg...) -> abs(f(arg...)), knots..., ; threaded=threaded)
norm_L1(f_data, ds...; threaded=true) = integrate(abs.(f_data), ds...; threaded=threaded)
norm_L2(f::Function, knots...; threaded=true) = sqrt(integrate((arg...) -> f(arg...)^2, knots...; threaded=threaded))
norm_L2(f_data, ds...; threaded=true) = integrate(f_data .^ 2, ds...; threaded=threaded)

function integrate_1d_serial_impl(f_data, dx::Number)
  res = @views 0.5 * dx * (sum(f_data[begin:end-1]) + sum(f_data[begin+1:end]))
  return res
end

function integrate_1d_serial_impl(f_data, dx)
  @assert length(f_data) == length(dx) + 1
  res = 0.5 * mapreduce(+, @view(f_data[begin:end-1]), @view(f_data[begin+1:end]), dx) do fL, fR, Δ
    return Δ * (fL + fR)
  end
  return res
end

function integrate_1d_serial_impl(f::Function, x::AbstractRange)
  dx = step(x)
  acc = f(x[begin]) + f(x[end])
  acc += 2 * mapreduce(f, +, @view(x[begin+1:end-1]); init=zero(acc))
  return 0.5 * dx * acc
end

function integrate_1d_threaded_impl(f::Function, x)
  res = @views 0.5 * mapreduce(+, x[begin:end-1], x[begin+1:end]) do xL, xR
    return (xR - xL) * (f(xL) + f(xR))
  end
  return res
end

function integrate_1d_threaded_impl(f_data, dx::Number)
  res = @views 0.5 * dx * tmapreduce(+, +, f_data[begin:end-1], f_data[begin+1:end])
  return res
end

function integrate_1d_threaded_impl(f_data, dx)
  @assert length(f_data) == length(dx) + 1
  res = 0.5 * tmapreduce(+, @view(f_data[begin:end-1]), @view(f_data[begin+1:end]), dx) do fL, fR, Δ
    return Δ * (fL + fR)
  end
  return res
end

function integrate_1d_threaded_impl(f::Function, x::AbstractRange)
  dx = step(x)
  acc = f(x[begin]) + f(x[end])
  acc += 2 * tmapreduce(f, +, @view(x[begin+1:end-1]); init=zero(acc))
  return 0.5 * dx * acc
end

function integrate_1d_threaded_impl(f::Function, x)
  res = @views 0.5 * tmapreduce(+, x[begin:end-1], x[begin+1:end]) do xL, xR
    return (xR - xL) * (f(xL) + f(xR))
  end
  return res
end

function integrate_2d_serial_impl(f_data, dx::Number, dy::Number)
  dA = dx * dy
  # corner terms (map abs first)
  acc = f_data[begin, begin] + f_data[end, end] + f_data[begin, end] + f_data[end, begin]
  @views begin
    # edge terms
    acc += 2 * sum(f_data[begin+1:end-1, begin])
    acc += 2 * sum(f_data[begin+1:end-1, end])
    acc += 2 * sum(f_data[begin, begin+1:end-1])
    acc += 2 * sum(f_data[end, begin+1:end-1])
    # bulk terms
    acc += 4 * sum(f_data[begin+1:end-1, begin+1:end-1])
  end
  return (acc / 4) * dA
end

function integrate_2d_serial_impl(f::Function, x::AbstractRange, y::AbstractRange)
  dA = step(x) * step(y)
  acc = f(x[begin], y[begin]) + f(x[end], y[begin]) + f(x[begin], y[end]) + f(x[end], y[end])
  acc += 2 * mapreduce(Fix2(f, y[begin]), +, x[begin+1:end-1])
  acc += 2 * mapreduce(Fix2(f, y[end]), +, x[begin+1:end-1])
  acc += 2 * mapreduce(Fix1(f, x[begin]), +, y[begin+1:end-1])
  acc += 2 * mapreduce(Fix1(f, x[end]), +, y[begin+1:end-1])
  i_idxs = axes(x, 1)
  j_idxs = axes(y, 1)
  idxs = CartesianIndex(2, 2):CartesianIndex(length(x) - 1, length(y) - 1)
  acc += 4 * mapreduce(+, idxs) do idx
    i = i_idxs[idx[1]]
    j = j_idxs[idx[2]]
    return f(x[i], y[j])
  end
  return (acc / 4) * dA
end

function integrate_2d_threaded_impl(f_data, dx::Number, dy::Number)
  dA = dx * dy
  # corner terms (map abs first)
  acc = f_data[begin, begin] + f_data[end, end] + f_data[begin, end] + f_data[end, begin]
  @views begin
    # edge terms
    acc += 2 * tmapreduce(+, +, f_data[begin+1:end-1, begin], f_data[begin+1:end-1, end])
    acc += 2 * tmapreduce(+, +, f_data[begin, begin+1:end-1], f_data[end, begin+1:end-1])
    # bulk terms
    acc += 4 * treduce(+, f_data[begin+1:end-1, begin+1:end-1])
  end
  return (acc / 4) * dA
end

function integrate_2d_threaded_impl(f::Function, x::AbstractRange, y::AbstractRange)
  dA = step(x) * step(y)
  acc = f(x[begin], y[begin]) + f(x[end], y[begin]) + f(x[begin], y[end]) + f(x[end], y[end])
  acc += 2 * tmapreduce(Fix2(f, y[begin]), +, x[begin+1:end-1])
  acc += 2 * tmapreduce(Fix2(f, y[end]), +, x[begin+1:end-1])
  acc += 2 * tmapreduce(Fix1(f, x[begin]), +, y[begin+1:end-1])
  acc += 2 * tmapreduce(Fix1(f, x[end]), +, y[begin+1:end-1])
  i_idxs = axes(x, 1)
  j_idxs = axes(y, 1)
  idxs = CartesianIndex(2, 2):CartesianIndex(length(x) - 1, length(y) - 1)
  acc += 4 * tmapreduce(+, idxs) do idx
    i = i_idxs[idx[1]]
    j = j_idxs[idx[2]]
    return f(x[i], y[j])
  end
  return (acc / 4) * dA
end

end
