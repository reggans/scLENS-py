poisson_dev_batch <- function(y,x) {
  if (is.null(x)) {
    n <- Matrix::colSums(y)
    pis <- Matrix::rowSums(y)/sum(y)
    mu <- crossprod(array(pis,dim=c(1,length(pis))),array(n,dim=c(1,length(n))))
    d <- 2 * (y * log(ifelse(y == 0, 1, y/mu)) - (y - mu))
    d[d<0] <- 0
    
    return(sqrt(d)*ifelse(y>mu,1,-1))
  } else {
    y1 <- lapply(unique(x),function(i) y[,x==i,drop=FALSE])
    n <- lapply(y1,Matrix::colSums)
    pis <- lapply(y1,function(data) Matrix::rowSums(data)/sum(data))
    mu <- lapply(1:length(y1),function(ind)
      crossprod(array(pis[[ind]],dim=c(1,length(pis[[ind]]))),
                array(n[[ind]],dim=c(1,length(n[[ind]])))))
    d <- lapply(1:length(y1),function(ind)
      2 * (y1[[ind]] * log(ifelse(y1[[ind]] == 0, 1, y1[[ind]]/mu[[ind]])) -
             (y1[[ind]] - mu[[ind]])))
    
    res <- array(0,dim=dim(y))
    rownames(res) <- rownames(y)
    colnames(res) <- colnames(y)
    for (ind in 1:length(y1)) {
      d[[ind]][d[[ind]]<0] <- 0
      res[,x==unique(x)[ind]] <- as.matrix(sqrt(d[[ind]])*
                                             ifelse(y1[[ind]]>mu[[ind]],1,-1))
    }
    
    return(res)
  }
}

reduce_dimension <- function(y,x,num_PCs) {
  pdev <- poisson_dev_batch(y,x)
  pdev <- t(scale(Matrix::t(pdev),scale=FALSE))
  PCs <- RSpectra::eigs_sym(as.matrix(tcrossprod(pdev)),k=num_PCs)
  projection <- t(crossprod(PCs$vectors,pdev))
  
  return(list(PCs, projection))
}

df <- read.csv("data/sim_Tcell/6083-4496.csv.gz")

cells <- df[, 1]
df_mat <- t(as.matrix(df[, 2:length(df)]))
df_mat <- as(df_mat, "CsparseMatrix")

batch <- rep("1",ncol(df_mat))

dev <- scry::devianceFeatureSelection(df_mat)
var.genes <- rownames(df_mat)[order(dev,decreasing=TRUE)[1:2500]]

reduce <- reduce_dimension(df_mat[var.genes,], NULL, 30)[[2]]

write.csv(reduce, "scshc_success.csv")
