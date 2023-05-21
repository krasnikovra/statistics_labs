x = (1 : size(input, 1)).'
residuals2 = input_int - tau2(1) - tau2(2) .* x

hold on
errorbar(mid(residuals2), rad(residuals2), "b")
plot(x, zeros(size(x, 1)), "r")
title("Regression with reducement residuals")
ylabel("mV")
xlabel("n")
xlim([1, size(residuals2, 1)])
s = [inf(residuals2).' sup(residuals2).']
ylim([min(s), max(s)])

print -dpdfcrop ../figures/regr_reduce_residuals.pdf

