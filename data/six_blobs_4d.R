n <- 300000

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
z <- rt(n, df = 10000)
w <- rt(n, df = 10000)
x_res <- x
y_res <- y
z_res <- z
w_res <- w

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
z <- rt(n, df = 10000)
x_res <- c(x_res, x + 10)
y_res <- c(y_res, y + 10)
z_res <- c(z_res, z + 10)
w_res <- c(w_res, w + 10)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
z <- rt(n, df = 10000)
x_res <- c(x_res, x - 10)
y_res <- c(y_res, y - 10)
z_res <- c(z_res, z - 10)
w_res <- c(w_res, w - 10)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
z <- rt(n, df = 10000)
x_res <- c(x_res, x - 10)
y_res <- c(y_res, y - 10)
z_res <- c(z_res, z + 10)
w_res <- c(w_res, w + 10)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
z <- rt(n, df = 10000)
x_res <- c(x_res, x + 10)
y_res <- c(y_res, y + 10)
z_res <- c(z_res, z - 10)
w_res <- c(w_res, w - 10)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
z <- rt(n, df = 10000)
x_res <- c(x_res, x + 10)
y_res <- c(y_res, y - 10)
z_res <- c(z_res, z + 10)
w_res <- c(w_res, w - 10)


res <- cbind.data.frame(x_res, y_res, z_res, w_res)

write.table(res[sample(1:nrow(res)), ], file = "six_blobs_4d.csv", row.names = FALSE, col.names = FALSE)