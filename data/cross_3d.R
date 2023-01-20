n <- 300000

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
z <- rt(n, df = 10000)
x_res <- x
y_res <- y
z_res <- z

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
z <- rt(n, df = 10000)
x_res <- c(x_res, x + 5)
y_res <- c(y_res, y + 5)
z_res <- c(z_res, z + 5)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
z <- rt(n, df = 10000)
x_res <- c(x_res, x - 5)
y_res <- c(y_res, y - 5)
z_res <- c(z_res, z - 5)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
z <- rt(n, df = 10000)
x_res <- c(x_res, x - 5)
y_res <- c(y_res, y + 5)
z_res <- c(z_res, z + 5)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
z <- rt(n, df = 10000)
x_res <- c(x_res, x + 5)
y_res <- c(y_res, y - 5)
z_res <- c(z_res, z - 5)

res <- cbind.data.frame(x_res, y_res, z_res)

write.table(res[sample(1:nrow(res)), ], file = "cross_3d.csv", row.names = FALSE, col.names = FALSE)