using SimpleIntegration

using Test
using StaticArrays

@testset "SimpleIntegration.jl" verbose = true begin
  knots = [2, 11, 101, 1001, 10001, 100001]
  @testset "1D-Trapezoidal" begin
    funcs = [x -> 3 * x^2, x -> sqrt(x), x -> 1 / x, x -> x]
    Funcs = [x -> x^3, x -> 2 / 3 * sqrt(x^3), x -> log(abs(x)), x -> x^2 / 2]
    domains = [
      [(0.0, 1.0), (-1.0, 1.0), (1.0, 10)],
      [(0.0, 1.0), (1.0, 10.0), (1.0, 10)],
      [(1.0, 2.0), (1.0, 10.0), (1.0, 100.0)],
      [(0.0, 1.0), (-1.0, 1.0), (1.0, 10)],
    ]

    for (f, F, dmn) ∈ zip(funcs, Funcs, domains)
      for d ∈ dmn
        Igral = F(d[2]) - F(d[1])
        igral_via_array = map(knots) do nx
          pts = range(d...; length=nx)
          data = f.(pts)
          dx = step(pts)
          return integrate(data, dx; threaded=false)
        end
        igral_via_fn = map(knots) do nx
          pts = range(d...; length=nx)
          return integrate(f, pts; threaded=false)
        end
        igral_via_array_t = map(knots) do nx
          pts = range(d...; length=nx)
          data = f.(pts)
          dx = step(pts)
          return integrate(data, dx)
        end
        igral_via_fn_t = map(knots) do nx
          pts = range(d...; length=nx)
          return integrate(f, pts)
        end

        # @show Igral
        # @show igral_via_fn
        # @show igral_via_array
        @test all(isapprox.(igral_via_fn, igral_via_array; atol=1.0e-8))
        @test all(isapprox.(igral_via_fn_t, igral_via_array_t; atol=1.0e-8))
        @test all(isapprox.(igral_via_fn, igral_via_array_t; atol=1.0e-8))
        @test all(isapprox.(igral_via_fn_t, igral_via_array; atol=1.0e-8))
        # relax approximate equality here, we just want to test for possible convergence
        @test any(v -> isapprox(Igral, v; atol=1.0e-6), igral_via_array)
      end
    end
  end

  steps2d = [2, 11, 101, 1001]
  @testset "2D-Trapezoidal" begin
    funcs = [(x, y) -> x^2 + y^2, (x, y) -> sin(2 * x) * cos(y / 2)]
    domains = [
      [((0.0, 0.0), (1.0, 1.0)), ((-10.0, -10.0), (10.0, 10.0))],
      [((-π, π), (-π, π))]
    ]
    for (f, dmn) ∈ zip(funcs, domains)
      for (dmn_x, dmn_y) ∈ dmn
        igral_via_array = map(steps2d) do nx
          pts_x = range(dmn_x...; length=nx)
          pts_y = range(dmn_y...; length=nx)
          data = f.(pts_x, pts_y')
          return integrate(data, step(pts_x), step(pts_y); threaded=false)
        end
        igral_via_array_t = map(steps2d) do nx
          pts_x = range(dmn_x...; length=nx)
          pts_y = range(dmn_y...; length=nx)
          data = f.(pts_x, pts_y')
          return integrate(data, step(pts_x), step(pts_y))
        end
        igral_via_fn = map(steps2d) do nx
          pts_x = range(dmn_x...; length=nx)
          pts_y = range(dmn_y...; length=nx)
          return integrate(f, pts_x, pts_y; threaded=false)
        end
        igral_via_fn_t = map(steps2d) do nx
          pts_x = range(dmn_x...; length=nx)
          pts_y = range(dmn_y...; length=nx)
          return integrate(f, pts_x, pts_y)
        end
        @test all(isapprox.(igral_via_array, igral_via_array_t; atol=1.0e-8))
        @test all(isapprox.(igral_via_array, igral_via_fn; atol=1.0e-8))
        @test all(isapprox.(igral_via_fn, igral_via_fn_t; atol=1.0e-8))
        @test all(isapprox.(igral_via_array_t, igral_via_fn_t; atol=1.0e-8))
      end
    end
  end

  @testset "L1-Norm" begin
    funcs = [
      x -> x,
      x -> x^2,
      x -> SVector(sin(x) / x, x * cos(x))
    ]
    norm1s = [
      x -> abs(x),
      x -> x^2,
      x -> sum(abs, SVector(sin(x) / x, x * cos(x)))
    ]

    domains = [
      [(0.0, 1.0), (-1.0, 1.0), (1.0, 10)],
      [(0.0, 1.0), (1.0, 10.0), (1.0, 10)],
      [(1.0, 2.0), (1.0, 10.0), (1.0, 100.0)],
      [(0.0, 1.0), (-1.0, 1.0), (1.0, 10)],
    ]

    for (f, f2, dmn) ∈ zip(funcs, norm1s, domains)
      for d ∈ dmn
        l1 = map(knots) do nx
          pts_x = range(d...; length=nx)
          return norm_L1(f, pts_x)
        end
        igral = map(knots) do nx
          pts_x = range(d...; length=nx)
          val = integrate(f2, pts_x)
          return val
        end
        @test all(isapprox.(l1, igral; atol=1.0e-8))
      end
    end
  end

  @testset "L2-Norm" begin
    funcs = [x -> x, x -> x^2, x -> SVector(sin(x) / x, x * cos(x))]
    norm2s = [x -> x^2, x -> x^4, x -> (sin(x) / x)^2 + (x * cos(x))^2]

    domains = [
      [(0.0, 1.0), (-1.0, 1.0), (1.0, 10)],
      [(0.0, 1.0), (1.0, 10.0), (1.0, 10)],
      [(1.0, 2.0), (1.0, 10.0), (1.0, 100.0)],
      [(0.0, 1.0), (-1.0, 1.0), (1.0, 10)],
    ]

    for (f, f2, dmn) ∈ zip(funcs, norm2s, domains)
      for d ∈ dmn
        l2 = map(knots) do nx
          pts_x = range(d...; length=nx)
          return norm_L2(f, pts_x)
        end
        igral = map(knots) do nx
          pts_x = range(d...; length=nx)
          val = integrate(f2, pts_x)
          return sqrt(val)
        end
        @test all(isapprox.(l2, igral; atol=1.0e-8))
      end
    end
  end
end
