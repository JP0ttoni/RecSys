using CSV
using DataFrames
using Statistics
using LinearAlgebra

# =========================================================
# CARREGAR DATASET MOVIELENS 100K
# =========================================================

df = CSV.read(
    "u.data",
    DataFrame,
    delim = '\t',
    header = ["user", "item", "rating", "timestamp"]
)

# =========================================================
# CRIAR MATRIZ USUÁRIO-ITEM
# =========================================================

n_users = maximum(df.user)
n_items = maximum(df.item)

# matriz de ratings
R = fill(0.0, n_users, n_items)

for row in eachrow(df)
    R[row.user, row.item] = row.rating
end

# =========================================================
# FUNÇÃO: COSSENO AJUSTADO
# =========================================================

function adjusted_cosine_similarity(u, v)

    # itens avaliados pelos dois usuários
    common_items = (u .> 0) .& (v .> 0)

    # sem itens em comum
    if sum(common_items) == 0
        return 0.0
    end

    # pega apenas itens em comum
    u_common = u[common_items]
    v_common = v[common_items]

    # médias dos usuários
    mean_u = mean(u_common)
    mean_v = mean(v_common)

    # centralização
    u_adjusted = u_common .- mean_u
    v_adjusted = v_common .- mean_v

    # denominador
    denominator = norm(u_adjusted) * norm(v_adjusted)

    if denominator == 0
        return 0.0
    end

    # similaridade
    similarity = dot(u_adjusted, v_adjusted) / denominator

    return similarity
end

# =========================================================
# EXEMPLO DE USO
# =========================================================

user1 = 1
user2 = 2

similarity = adjusted_cosine_similarity(
    R[user1, :],
    R[user2, :]
)

println("Similaridade entre usuário $user1 e usuário $user2:")
println(similarity)