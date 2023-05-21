x = 1:size(input,1).'
[irproblem] = ir_problem([x.^0; x].', input, max(w2) * eps)

vertices = ir_beta2poly(irproblem)

b_int = ir_outer(irproblem)

hold on
x = vertices(:, 1)
y = vertices(:, 2)
ir_plotbeta(irproblem)
ir_plotrect(b_int, "r")
title("Information set")
xlim([min(x) - 1e-5, max(x) + 1e-5])
ylim([min(y) - 1e-7, max(y) + 1e-7])
xlabel("\\beta_0")
ylabel("\\beta_1")

print -dpdfcrop ../figures/inform_set.pdf

