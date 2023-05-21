[tau1, w1, yint1] = DataLinearModel(input, eps)
sum_w1 = sum(w1)

hold on
errorbar(input, eps, "b")
x = [1, size(input, 1)]
plot(x, tau1(1) + tau1(2) .* x, "r")
title("Data simple regression")
xlabel("n")
ylabel("mV")
xlim([1, size(input, 1)])
ylim([input(1) - eps * w1(1), input(end) + eps * w1(end)])

print -dpdfcrop ../figures/data_regr.pdf

