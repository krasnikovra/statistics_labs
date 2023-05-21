x = (1 : size(input, 1)).'

residuals1 = input_int - tau1(1) - tau1(2) .* x

hold on
errorbar(mid(residuals1), rad(residuals1), "b")
plot(x, zeros(size(x, 1)), "r")
title("Regression residuals")
ylabel("mV")
xlabel("n")
xlim([1, size(residuals1, 1)])
s = [inf(residuals1).' sup(residuals1).']
ylim([min(s), max(s)])

print -dpdfcrop ../figures/regr_residuals.pdf

