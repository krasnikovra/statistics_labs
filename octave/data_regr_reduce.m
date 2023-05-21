[tau2, w2, yint2] = DataLinearModelZ(input, eps)
sum_w2 = sum(w2)

hold on
errorbar(input, eps, "b")
x = [1, size(input, 1)]
errorbar(input, eps * w2, "y")
plot(x, tau2(1) + tau2(2) .* x, "r")
title("Data regression with reducement of intervals")
xlabel("n")
ylabel("mV")
xlim([1, size(input, 1)])
ylim([input(1) - eps * w2(1), input(end) + eps * w2(end)])

print -dpdfcrop ../figures/data_regr_reduce.pdf

