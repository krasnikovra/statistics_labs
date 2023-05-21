setenv ("OCTAVE_LATEX_DEBUG_FLAG", "1")

alpha = sum_w2 / 100

hold on

plot(w1)
plot(w2)
plot(alpha .* ones(size(w2, 1)), "k--")

title("\\omega_1 and \\omega_2 values")
legend("\\omega_1", "\\omega_2")
ylabel("\\omega_i")
xlabel("i")
text(5, alpha + 0.05, "\\alpha");

ylim([0, max(w2) + 0.5])

print -dpdfcrop ../figures/omegas.pdf

