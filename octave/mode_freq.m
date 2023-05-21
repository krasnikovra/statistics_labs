[mode1, modefreq1, freqs1, Ss1] = imodeR([inf(residuals1) sup(residuals1)])
[mode2, modefreq2, freqs2, Ss2] = imodeR([inf(residuals2) sup(residuals2)])

hold on

plot(Ss1(1:end-1), freqs1, "r")
plot(Ss2(1:end-1), freqs2, "b")

title("Mode frequencies of residuals")
xlabel("mV")
ylabel("\\mu")
legend("1st model \\omega \\geq 1", "2nd model \\omega \\geq 0")
xlim([min(Ss1) max(Ss1)])

print -dpdfcrop ../figures/mode_freq.pdf

