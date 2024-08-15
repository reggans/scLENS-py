library(Seurat)

df <- read.csv("data/sim_Tcell/6083-4496.csv.gz
               ")

cells <- df[, 1]
df_mat <- t(as.matrix(df[, 2:length(df)]))
df_mat <- as(df_mat, "CsparseMatrix")

seu <- CreateSeuratObject(df_mat)
seu <- NormalizeData(seu)

seu <- FindVariableFeatures(object = seu, selection.method = "vst", nfeatures = 2000,
                           loess.span = 0.3, clip.max = "auto",
                           num.bin = 20, binning.method = "equal_width", verbose = F)

all.genes <- rownames(x = seu)
seu <- ScaleData(object = seu, features = all.genes)
seu <- RunPCA(object = seu, features = VariableFeatures(object = seu), npcs=30)

write.csv(seu@reductions$pca@cell.embeddings, "success_multik.csv")
