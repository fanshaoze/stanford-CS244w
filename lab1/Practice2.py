import json
import matplotlib.pyplot as plt
_seed = 5
plots = json.load(open(f'practice_feature_{_seed}.json'))
plots_sig = json.load(open(f'practice_sig_{_seed}.json'))
plots_diff = [plots_sig[1][i]-plots[1][i] for i in range(len(plots_sig[1]))]
plt.plot(plots[0], plots_diff)
plt.savefig(f"practice_diff_{_seed}.png", dpi=300, format="png")
plt.close()
