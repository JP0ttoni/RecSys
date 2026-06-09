using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Random

# =========================================================
# CARREGAR O ARQUIVO DE RATINGS
# =========================================================

ratings = CSV.read(
    "ml-100k/u.data",
    DataFrame,
    delim = '\t',
    header = ["user_id", "movie_id", "rating", "timestamp"]
)

# =========================================================
# CRIAR MATRIZ USUÁRIO-ITEM
# =========================================================

n_users = maximum(ratings.user_id)
n_items = maximum(ratings.movie_id)

R = fill(0.0, n_users, n_items)

for row in eachrow(ratings)
    R[row.user_id, row.movie_id] = row.rating
end

# =========================================================
# COSSENO AJUSTADO ENTRE DOIS USUÁRIOS
# =========================================================

function adjusted_cosine_similarity(u, v)

    global common_items = (u .> 0) .& (v .> 0)

    if sum(common_items) == 0
        return 0.0
    end

    mean_u = mean(u[u .> 0])
    mean_v = mean(v[v .> 0])

    u_adjusted = u[common_items] .- mean_u
    v_adjusted = v[common_items] .- mean_v

    denominator = norm(u_adjusted) * norm(v_adjusted)

    if denominator == 0
        return 0.0
    end

    return dot(u_adjusted, v_adjusted) / denominator
end

# =========================================================
# ESCOLHER 2 USUÁRIOS ALEATÓRIOS
# =========================================================

target_user = 1#rand(1:943)

best_user = -1
best_similarity = -Inf

for other_user in 1:n_users

    if other_user == target_user
        continue
    end

    common_items =
        (R[target_user,:] .> 0) .&
        (R[other_user,:] .> 0)

    common_count = sum(common_items)

    if common_count < 10
        continue
    end

    sim = adjusted_cosine_similarity(
        R[target_user,:],
        R[other_user,:]
    )

    if sim > best_similarity
        global best_similarity = sim
        global best_user = other_user
        global best_common = common_count
    end
end

println("Usuário mais parecido com $target_user")
println("Usuário: $best_user")
println("Filmes em comum: $best_common")
println("Similaridade: $best_similarity")