using Pkg
# Tenta carregar o pacote, se falhar, instala automaticamente
try
    using UnicodePlots
catch
    Pkg.add("UnicodePlots")
    using UnicodePlots
end

using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Random

function load_movielens_latest_small(filepath)
    df = CSV.read(filepath, DataFrame)
    rename!(df, :userId => :user_id, :movieId => :item_id)
    dropmissing!(df, [:user_id, :item_id, :rating])
    return df[:, [:user_id, :item_id, :rating]]
end

function create_utility_matrix(df, num_users, num_items)
    R = zeros(Float64, num_users, num_items)
    for row in eachrow(df)
        R[row.user_id, row.item_id] = Float64(row.rating)
    end
    return R
end

function adjusted_cosine_similarity(R)
    num_users, num_items = size(R)
    user_means = zeros(num_users)
    for u in 1:num_users
        rated = R[u, :] .> 0
        user_means[u] = any(rated) ? mean(R[u, rated]) : 0.0
    end
    
    R_centered = zeros(num_users, num_items)
    for u in 1:num_users
        for i in 1:num_items
            if R[u, i] > 0
                R_centered[u, i] = R[u, i] - user_means[u]
            end
        end
    end
    
    S = zeros(num_items, num_items)
    norm_col = [norm(R_centered[:, i]) for i in 1:num_items]
    
    for i in 1:num_items
        for j in i:num_items
            if norm_col[i] > 0 && norm_col[j] > 0
                dot_prod = dot(R_centered[:, i], R_centered[:, j])
                sim = dot_prod / (norm_col[i] * norm_col[j])
                S[i, j] = sim
                S[j, i] = sim
            end
        end
    end
    return S, user_means
end

function predict_item_based(R, S, user_means, u, i, k=30)
    rated_items = findall(R[u, :] .> 0)
    if isempty(rated_items)
        return user_means[u]
    end
    
    sims = S[i, rated_items]
    p = sortperm(sims, rev=true)
    top_k = p[1:min(k, length(p))]
    
    chosen_items = rated_items[top_k]
    chosen_sims = sims[top_k]
    
    pos_idx = chosen_sims .> 0
    if !any(pos_idx)
        return user_means[u]
    end
    
    chosen_items = chosen_items[pos_idx]
    chosen_sims = chosen_sims[pos_idx]
    
    num = sum(chosen_sims .* R[u, chosen_items])
    den = sum(abs.(chosen_sims))
    
    return den > 0 ? num / den : user_means[u]
end

function train_regularized_svd(R, train_indices, num_users, num_items; factors=20, lr=0.005, lambda=0.02, epochs=20)
    P = randn(factors, num_users) .* 0.1
    Q = randn(factors, num_items) .* 0.1
    
    for epoch in 1:epochs
        for idx in train_indices
            u, i, rating = idx[1], idx[2], idx[3]
            pred = dot(P[:, u], Q[:, i])
            err = rating - pred
            
            p_u_old = P[:, u]
            P[:, u] .+= lr .* (err .* Q[:, i] .- lambda .* P[:, u])
            Q[:, i] .+= lr .* (err .* p_u_old .- lambda .* Q[:, i])
        end
    end
    return P, Q
end

function predict_svd(P, Q, u, i)
    pred = dot(P[:, u], Q[:, i])
    return clamp(pred, 1.0, 5.0)
end

function evaluate_models(df, num_users, num_items)
    Random.seed!(42)
    n = nrow(df)
    shuffled_indices = shuffle(1:n)
    train_size = floor(Int, 0.8 * n)
    
    train_df = df[shuffled_indices[1:train_size], :]
    test_df = df[shuffled_indices[train_size+1:end], :]
    
    R_train = create_utility_matrix(train_df, num_users, num_items)
    count_test = nrow(test_df)
    
    # Armazenamento para os gráficos
    k_vals = [10.0, 30.0, 50.0]
    rmse_k = Float64[]
    mae_k = Float64[]
    
    println("\n=============================================")
    println("1. PRINTS DE AVALIAÇÃO: ITEM-BASED")
    println("=============================================")
    println("Calculando matriz de similaridade (Cosseno Ajustado)...")
    S, user_means = adjusted_cosine_similarity(R_train)
    
    for k_test in [10, 30, 50]
        mae_ib, rmse_ib = 0.0, 0.0
        for row in eachrow(test_df)
            u, i, actual = row.user_id, row.item_id, row.rating
            pred = predict_item_based(R_train, S, user_means, u, i, k_test)
            
            err = actual - pred
            mae_ib += abs(err)
            rmse_ib += err^2
        end
        mae_ib /= count_test
        rmse_ib = sqrt(rmse_ib / count_test)
        
        push!(mae_k, mae_ib)
        push!(rmse_k, rmse_ib)
        
        # PRINT TEXTUAL SOLICITADO
        println("-> RESULTADO [k = $k_test]: MAE = $(round(mae_ib, digits=4)) | RMSE = $(round(rmse_ib, digits=4))")
    end
    
    factors_vals = [5.0, 15.0, 30.0]
    rmse_factors = Float64[]
    mae_factors = Float64[]
    
    println("\n=============================================")
    println("2. PRINTS DE AVALIAÇÃO: REGULARIZED SVD")
    println("=============================================")
    train_tuples = [(row.user_id, row.item_id, Float64(row.rating)) for row in eachrow(train_df)]
    
    lr = 0.007
    lambda = 0.04
    epochs = 20
    
    best_preds_svd = Float64[]
    actual_labels = Int[]
    
    for factors_test in [5, 15, 30]
        println("Treinando SVD com Fatores = $factors_test...")
        P, Q = train_regularized_svd(R_train, train_tuples, num_users, num_items, 
                                     factors=factors_test, lr=lr, lambda=lambda, epochs=epochs)
        
        mae_svd, rmse_svd = 0.0, 0.0
        preds_atual = Float64[]
        
        for row in eachrow(test_df)
            u, i, actual = row.user_id, row.item_id, row.rating
            pred = predict_svd(P, Q, u, i)
            
            err = actual - pred
            mae_svd += abs(err)
            rmse_svd += err^2
            
            if factors_test == 15
                push!(preds_atual, pred)
                push!(actual_labels, actual >= 4.0 ? 1 : 0)
            end
        end
        if factors_test == 15
            append!(best_preds_svd, preds_atual)
        end
        
        mae_svd /= count_test
        rmse_svd = sqrt(rmse_svd / count_test)
        
        push!(mae_factors, mae_svd)
        push!(rmse_factors, rmse_svd)
        
        # PRINT TEXTUAL SOLICITADO
        println("-> RESULTADO [Fatores = $factors_test]: MAE = $(round(mae_svd, digits=4)) | RMSE = $(round(rmse_svd, digits=4))\n")
    end
    
    # =========================================================
    # VISUALIZAÇÕES ADICIONAIS NO TERMINAL (UnicodePlots)
    # =========================================================
    println("\n=============================================")
    println("3. VISUALIZAÇÕES GRÁFICAS NO TERMINAL")
    println("=============================================")
    
    println("\n[Gráfico 1] Evolução do RMSE - Item-Based (K):")
    display(lineplot(k_vals, rmse_k, xlabel="Valor de K", ylabel="RMSE", color=:blue))
    
    println("\n[Gráfico 2] Evolução do RMSE - SVD (Fatores):")
    display(lineplot(factors_vals, rmse_factors, xlabel="Fatores", ylabel="RMSE", color=:red))

    # Curva ROC Adaptada
    thresholds = range(1.0, 5.0, length=20)
    tpr_vals = Float64[]
    fpr_vals = Float64[]
    
    for t in thresholds
        tp = sum((best_preds_svd .>= t) .& (actual_labels .== 1))
        fp = sum((best_preds_svd .>= t) .& (actual_labels .== 0))
        fn = sum((best_preds_svd .< t) .& (actual_labels .== 1))
        tn = sum((best_preds_svd .< t) .& (actual_labels .== 0))
        
        tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
        fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0.0
        
        push!(tpr_vals, tpr)
        push!(fpr_vals, fpr)
    end
    
    p_sort = sortperm(fpr_vals)
    println("\n[Gráfico 3] Curva ROC - Regularized SVD (Fatores=15):")
    display(lineplot(fpr_vals[p_sort], tpr_vals[p_sort], xlabel="FPR", ylabel="TPR", color=:green))
    println("=============================================")
end

# --- Bloco de Execução Principal ---
filepath = "./ml-latest-small/ratings.csv" 

if isfile(filepath)
    df = load_movielens_latest_small(filepath)
    
    user_map = Dict(id => idx for (idx, id) in enumerate(unique(df.user_id)))
    item_map = Dict(id => idx for (idx, id) in enumerate(unique(df.item_id)))
    
    df.user_id = [user_map[id] for id in df.user_id]
    df.item_id = [item_map[id] for id in df.item_id]
    
    num_users = length(user_map)
    num_items = length(item_map)
    
    println("Dados carregados com sucesso!")
    println("Utilizadores únicos: $num_users | Filmes únicos: $num_items")
    
    evaluate_models(df, num_users, num_items)
else
    println("Erro: Não foi possível encontrar o arquivo no caminho especificado.")
end