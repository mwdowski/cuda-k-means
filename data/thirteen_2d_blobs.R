n <- 200000
x <- rt(n, df = 10000)
y <- rt(n, df = 10000)

x_res <- x
y_res <- y

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)

x_res <- c(x_res, x + 5)
y_res <- c(y_res, y + 5)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
x_res <- c(x_res, x - 5)
y_res <- c(y_res, y + 5)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
x_res <- c(x_res, x + 5)
y_res <- c(y_res, y - 5)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)

x_res <- c(x_res, x - 5)
y_res <- c(y_res, y - 5)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
x_res <- c(x_res, x + 10)
y_res <- c(y_res, y + 10)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
x_res <- c(x_res, x - 10)
y_res <- c(y_res, y - 10)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
x_res <- c(x_res, x + 10)
y_res <- c(y_res, y - 10)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
x_res <- c(x_res, x - 10)
y_res <- c(y_res, y + 10)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
x_res <- c(x_res, x - 10)
y_res <- c(y_res, y)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
x_res <- c(x_res, x + 10)
y_res <- c(y_res, y)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
x_res <- c(x_res, x)
y_res <- c(y_res, y - 10)

x <- rt(n, df = 10000)
y <- rt(n, df = 10000)
x_res <- c(x_res, x)
y_res <- c(y_res, y + 10)

res <- cbind.data.frame(x_res, y_res)

# plot(x_res, y_res)

write.table(sample(res), file = "thirteen_2d_blobs.csv", row.names = FALSE, col.names = FALSE)
