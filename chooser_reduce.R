source("R/pipeline.R")
library(Seurat)
library(ggplot2)
`%>%` <- magrittr::`%>%`
library(sctransform)

df <- read.csv("../data/sim_Tcell/6083-4496.csv.gz")

cells <- df[, 1]
df_mat <- t(as.matrix(df[, 2:length(df)]))
df_mat <- as(df_mat, "CsparseMatrix")

seu <- CreateSeuratObject(df_mat)
seu <- SCTransform(seu)
seu <- RunPCA(seu, npcs=100)

write.csv(seu@reductions$pca@cell.embeddings, "../../../scLENS/success_chooser.csv")

# Define the number of PCs to use, and which assay and reduction to use.
# We recommend testing a broad range of resolutions
# For more on picking the correct number of PCs, see:
# https://satijalab.org/seurat/v3.1/pbmc3k_tutorial.html
npcs <- 50 # Default value in Seurat
resolutions <- seq(0.1, 2, 0.1)
assay <- "RNA"
reduction <- "pca"
filename <- gsub("/", "_", "success_chooser")
filename <- gsub(".csv.gz", "", filename)
results_path <- sprintf("../R_comparison_clusters/ChooseR/%s/", filename)

if (!dir.exists(results_path)){
  dir.create(results_path)
}

# Run pipeline
for (res in resolutions) {
  message(paste0("Clustering ", res, "..."))
  message("\tFinding ground truth...")
  
  # "Truths" will be stored at glue::glue("{reduction}.{assay}_res.{res}")
  seu <- find_clusters(
    seu,
    reduction = reduction,
    assay = assay,
    resolution = res,
    npcs = npcs
  )
  clusters <- seu[[glue::glue("{reduction}.{assay}_res.{res}")]]
  
  # Now perform iterative, sub-sampled clusters
  results <- multiple_cluster(
    seu,
    n = 10,
    size = 0.8,
    npcs = npcs,
    res = res,
    reduction = reduction,
    assay = assay
  )
  
  # Now calculate the co-clustering frequencies
  message(paste0("Tallying ", res, "..."))
  # This is the more time efficient vectorisation
  # However, it exhausts vector memory for (nearly) all datasets
  # matches <- purrr::map(columns, find_matches, df = results)
  # matches <- purrr::reduce(matches, `+`)
  columns <- colnames(dplyr::select(results, -cell))
  mtchs <- matrix(0, nrow = dim(results)[1], ncol = dim(results)[1])
  i <- 1 # Counter
  for (col in columns) {
    message(paste0("\tRound ", i, "..."))
    mtchs <- Reduce("+", list(
      mtchs,
      find_matches(col, df = results)
    ))
    i <- i + 1
  }
  
  message(paste0("Scoring ", res, "..."))
  mtchs <- dplyr::mutate_all(
    dplyr::as_tibble(mtchs),
    function(x) dplyr::if_else(Re(x) > 0, percent_match(x), 0)
  )
  
  # Now calculate silhouette scores
  message(paste0("Silhouette ", res, "..."))
  sil <- cluster::silhouette(
    x = as.numeric(as.character(unlist(clusters))),
    dmatrix = (1 - as.matrix(mtchs))
  )
  saveRDS(sil, paste0(results_path, "silhouette_", res, ".rds"))
  
  # Finally, calculate grouped metrics
  message(paste0("Grouping ", res, "..."))
  grp <- group_scores(mtchs, unlist(clusters))
  saveRDS(grp, paste0(results_path, "frequency_grouped_", res, ".rds"))
  sil <- group_sil(sil, res)
  saveRDS(sil, paste0(results_path, "silhouette_grouped_", res, ".rds"))
}

# Save original data, with ground truth labels
saveRDS(seu, paste0(results_path, "clustered_data.rds"))

# Create silhouette plot
# Read in scores and calculate CIs
scores <- purrr::map(
  paste0(results_path, "silhouette_grouped_", resolutions, ".rds"),
  readRDS
)
scores <- dplyr::bind_rows(scores) %>%
  dplyr::group_by(res) %>%
  dplyr::mutate("n_clusters" = dplyr::n()) %>%
  dplyr::ungroup()
meds <- scores %>%
  dplyr::group_by(res) %>%
  dplyr::summarise(
    "boot" = list(boot_median(avg_sil)),
    "n_clusters" = mean(n_clusters)
  ) %>%
  tidyr::unnest_wider(boot)

writexl::write_xlsx(meds, paste0(results_path, "median_ci.xlsx"))

# Find thresholds
threshold <- max(meds$low_med)
choice <- as.character(
  meds %>%
    dplyr::filter(med >= threshold) %>%
    dplyr::arrange(n_clusters) %>%
    tail(n = 1) %>%
    dplyr::pull(res)
)

cls <- seu@meta.data$pca.RNA_res.2
write.csv(cls, "../../../scLENS/success_cls_chooser.csv")

