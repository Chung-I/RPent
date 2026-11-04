# First-menu decisions under each prior, against the class prior

| setting | prior | n | recommendation changed [95 % CI] | recommended row changed [95 % CI] | transitions |
|---|---|---|---|---|---|
| k10_A10 | gemma4_31b | 1045 | 0.216 [0.158, 0.278] | 0.256 [0.196, 0.318] | {"decline->commit": 17, "decline->probe": 61, "commit->probe": 111, "probe->commit": 8, "probe->decline": 20, "commit->decline": 9} |
| k10_A10 | generic | 1045 | 0.245 [0.181, 0.309] | 0.279 [0.212, 0.349] | {"decline->commit": 56, "decline->probe": 48, "commit->probe": 120, "probe->commit": 10, "probe->decline": 12, "commit->decline": 10} |
| k10_A10 | qwen38_27b | 1045 | 0.216 [0.159, 0.277] | 0.258 [0.196, 0.323] | {"decline->commit": 17, "decline->probe": 60, "commit->probe": 109, "probe->commit": 10, "probe->decline": 22, "commit->decline": 8} |
| k100_A3 | gemma4_31b | 1045 | 0.098 [0.061, 0.138] | 0.100 [0.064, 0.139] | {"commit->decline": 95, "commit->probe": 7} |
| k100_A3 | generic | 1045 | 0.077 [0.047, 0.111] | 0.084 [0.053, 0.121] | {"commit->decline": 70, "commit->probe": 10} |
| k100_A3 | qwen38_27b | 1045 | 0.099 [0.063, 0.137] | 0.101 [0.064, 0.142] | {"commit->decline": 95, "commit->probe": 8} |
| k100_A10 | gemma4_31b | 1045 | 0.166 [0.114, 0.218] | 0.175 [0.124, 0.230] | {"decline->probe": 9, "commit->decline": 60, "probe->decline": 36, "commit->probe": 62, "decline->commit": 6} |
| k100_A10 | generic | 1045 | 0.199 [0.150, 0.252] | 0.211 [0.158, 0.266] | {"decline->probe": 22, "decline->commit": 9, "commit->decline": 55, "probe->decline": 37, "commit->probe": 85} |
| k100_A10 | qwen38_27b | 1045 | 0.183 [0.128, 0.239] | 0.192 [0.144, 0.248] | {"commit->probe": 76, "commit->decline": 64, "probe->decline": 37, "decline->commit": 8, "decline->probe": 5, "probe->commit": 1} |
| k30_A3 | gemma4_31b | 1045 | 0.177 [0.126, 0.232] | 0.186 [0.134, 0.237] | {"commit->decline": 105, "commit->probe": 40, "decline->commit": 6, "probe->decline": 22, "decline->probe": 12} |
| k30_A3 | generic | 1045 | 0.174 [0.125, 0.227] | 0.183 [0.131, 0.236] | {"decline->probe": 12, "commit->decline": 106, "commit->probe": 41, "decline->commit": 3, "probe->decline": 20} |
| k30_A3 | qwen38_27b | 1045 | 0.188 [0.135, 0.248] | 0.196 [0.141, 0.255] | {"commit->decline": 118, "commit->probe": 42, "decline->commit": 3, "probe->decline": 23, "decline->probe": 10} |
| k30_A10 | gemma4_31b | 1045 | 0.200 [0.148, 0.255] | 0.211 [0.159, 0.268] | {"decline->probe": 46, "decline->commit": 7, "probe->decline": 38, "commit->decline": 17, "commit->probe": 98, "probe->commit": 3} |
| k30_A10 | generic | 1045 | 0.195 [0.146, 0.252] | 0.208 [0.155, 0.263] | {"decline->commit": 26, "decline->probe": 45, "probe->decline": 27, "commit->decline": 10, "commit->probe": 91, "probe->commit": 5} |
| k30_A10 | qwen38_27b | 1045 | 0.188 [0.141, 0.235] | 0.206 [0.159, 0.258] | {"decline->probe": 38, "probe->decline": 40, "decline->commit": 12, "commit->decline": 11, "commit->probe": 92, "probe->commit": 3} |
| k100_A30 | gemma4_31b | 1045 | 0.183 [0.135, 0.235] | 0.201 [0.153, 0.252] | {"decline->probe": 47, "probe->decline": 40, "commit->probe": 84, "commit->decline": 7, "decline->commit": 8, "probe->commit": 5} |
| k100_A30 | generic | 1045 | 0.184 [0.134, 0.233] | 0.195 [0.144, 0.249] | {"decline->commit": 25, "decline->probe": 50, "probe->decline": 28, "commit->decline": 7, "commit->probe": 78, "probe->commit": 4} |
| k100_A30 | qwen38_27b | 1045 | 0.186 [0.141, 0.233] | 0.205 [0.158, 0.257] | {"decline->probe": 44, "commit->probe": 90, "decline->commit": 12, "probe->decline": 41, "commit->decline": 2, "probe->commit": 5} |
