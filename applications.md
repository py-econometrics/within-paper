Labour — worker × firm (AKM)

Abowd, Kramarz & Margolis (1999), "High Wage Workers and High Wage Firms," Econometrica 67(2): 251–333. The origin of the two-way worker+firm FE decomposition; over one million French workers across 500,000+ firms. https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00020 Wiley Online Library
Card, Heining & Kline (2013), "Workplace Heterogeneity and the Rise of West German Wage Inequality," QJE 128(3): 967–1015. Fits additive worker and establishment FE across four sub-periods and attributes most of the inequality rise to plant premiums and assortative sorting. https://academic.oup.com/qje/article/128/3/967/1848785 Oxford Academic
Song, Price, Guvenen, Bloom & von Wachter (2019), "Firming Up Inequality," QJE 134(1): 1–50. Worker+firm decomposition on the near-universe of US matched employer–employee records, 1978–2013 — among the largest FE problems in the literature. https://academic.oup.com/qje/article-abstract/134/1/1/5144785 Oxford Academic

Trade — gravity (exporter×time, importer×time, pair FE)

Correia, Guimarães & Zylkin (2020), "Fast Poisson estimation with high-dimensional fixed effects," Stata Journal 20(1): 95–115 (ppmlhdfe). The PPML-HDFE workhorse; modified IRLS for fast estimation with multiple HDFE, since structural gravity routinely needs three sets of fixed effects. Direct methodological sibling to your solver. https://journals.sagepub.com/doi/10.1177/1536867X20909691 (preprint: https://arxiv.org/abs/1903.01690) Sage Journals
Silva & Tenreyro (2006), "The Log of Gravity," Review of Economics and Statistics 88(4): 641–658. The paper that made PPML the default gravity estimator (and motivates Poisson-FE solvers). https://doi.org/10.1162/rest.88.4.641
Also relevant: Larch, Wanner, Yotov & Zylkin (2019), "Currency Unions and Trade: A PPML Re-assessment with High-Dimensional Fixed Effects," Oxford Bulletin of Economics and Statistics 81(3) — iterative PPML built to absorb three-way gravity FE on large panels. (I have the citation but didn't capture a clean URL this pass — say the word and I'll grab it.)

Innovation — inventor × patent / city

Moretti (2021), "The Effect of High-Tech Clusters on the Productivity of Top Inventors," AER 111(10): 3328–3375. Longitudinal data on 109,846 inventors with inventor FE, identified off inventors moving across cities — the cleanest inventor-mover FE design. https://www.aeaweb.org/articles?id=10.1257/aer.20191277 (note: there's a published Wiebe comment flagging coding/identification issues in the event-study/IV, https://www.aeaweb.org/articles?id=10.1257/aer.20231415) American Economic Association
Bell, Chetty, Jaravel, Petkova & Van Reenen (2019), "Who Becomes an Inventor in America?", QJE 134(2): 647–713. 1.2 million inventors from patent records linked to tax records; FE-heavy specifications on a large patent–administrative linkage (exposure design rather than a pure two-way decomposition, but the data scale is the point). https://academic.oup.com/qje/article/134/2/647/5218522 NBER

Industrial organization

DellaVigna & Gentzkow (2019), "Uniform Pricing in US Retail Chains," QJE 134(4): 2011–2084. Nielsen scanner data with store/chain × product × time FE; documents near-uniform pricing across stores despite wide demand variation, attributing it to managerial fixed costs. https://www.nber.org/papers/w23996 Economics Job Market Rumors
Adjacent FE-intensive IO classes worth a sentence each in your intro: scanner-data demand estimation with brand×market×time FE; patent-examiner-leniency designs (thousands of examiner FE used as instruments); and buyer–seller FE in production-network data. Happy to pull specific cites (e.g., Sampat–Williams, Farre-Mensa–Hegde–Ljungqvist) if you want them.

Business / tech economics

Bertrand & Schoar (2003), "Managing with Style: The Effect of Managers on Firm Policies," QJE 118(4): 1169–1208. Manager–firm matched panel tracking executives across firms; manager fixed effects explain heterogeneity in investment, financial, and organizational decisions — the canonical "manager FE" / mover design, and the template for CEO/founder-effect work. https://academic.oup.com/qje/article-abstract/118/4/1169/1925095 Oxford Academic
Tech framing for the section: high-dimensional FE are standard in digital-platform/marketplace panels (user, item/seller, and time effects) and in regression-adjusted experiment analysis at industry data scale — precisely where solver throughput, not identification, is the binding constraint. This is a clean motivating paragraph; I can source concrete platform-economics examples if you want named cites rather than a framing sentence.