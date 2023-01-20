n <- 500000
x <- rt(n, df = 10000)
y <- rt(n, df = 10000)

x_res <- x
y_res <- y

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)

x_res <- c(x_res, x + 5)
y_res <- c(y_res, y + 5)

x <- rt(n, df = 100000)
y <- rt(n, df = 100000)

x_res <- c(x_res, (x - 2) * 1.2)
y_res <- c(y_res, (y + 6) * 1.2)

res <- cbind.data.frame(x_res, y_res)

write.table(res[sample(1:nrow(res)), ], file = "three_2d_blobs.csv", row.names = FALSE, col.names = FALSE)
