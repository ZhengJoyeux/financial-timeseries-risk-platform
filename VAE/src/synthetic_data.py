"""Only for DEMONSTRATION: 80 simulated companies, not real accounting records."""
import numpy as np
from .schema import KEYS


def build_simulated_companies(seed=20261023, n_companies=80, n_risk=16):
    if (n_companies, n_risk) != (80, 16):
        raise ValueError('第一阶段固定模拟80家，其中16家有风险。')
    r = np.random.default_rng(seed)
    risk_ids = set(r.choice(np.arange(1, 81), size=n_risk, replace=False).tolist())
    rows, labels = [], []
    idx = {name: j for j, name in enumerate(KEYS)}
    for firm_id in range(1, 81):
        risk = firm_id in risk_ids
        # Scenario diversity: varying sector-like turnover, size, leverage, margins, growth, risk manifestations.
        size = float(np.exp(r.uniform(np.log(6500), np.log(420000))))
        turnover = r.uniform(.45, 2.6)
        base_margin = r.uniform(.13, .46)
        base_opex = r.uniform(.08, .26)
        debt_ratio = r.uniform(.23, .77) + (.08 if risk else 0)
        growth = r.uniform(-.07, .15) + (r.uniform(-.06, .03) if risk else 0)
        flow_volatility = r.uniform(.025, .095)
        disturbance = r.choice(['decline_margin', 'high_leverage', 'cash_crunch', 'sales_fall']) if risk else 'none'
        stress_magnitude = r.uniform(.12, .42) if risk else r.uniform(0, .08)
        # Some healthy companies also show partial distress signals to avoid a trivially separable label.
        if not risk and r.random() < .23:
            stress_magnitude = r.uniform(.06, .18)
        company = np.zeros((20, len(KEYS)), dtype=np.float64)
        prev_working_capital = None
        asset_noise = 0.0
        for q in range(20):
            year_frac = q / 4
            late_stress = max(0.0, (q - int(r.integers(10, 15))) / 7)
            stress = min(1.0, late_stress) * stress_magnitude
            asset_noise = .7 * asset_noise + r.normal(0, .018)
            assets = max(450.0, size * np.exp(growth * year_frac + asset_noise) * (1-.15*stress))
            liabilities_ratio = debt_ratio + (stress * (.35 if disturbance == 'high_leverage' else .14))
            liabilities_ratio += r.normal(0, .016)
            liabilities = assets * np.clip(liabilities_ratio, .12, 1.16)
            current_assets = assets * np.clip(r.uniform(.32, .71) + r.normal(0, .03) - .11*stress, .16, .85)
            current_liabilities = liabilities * np.clip(r.uniform(.34, .68) + .07*stress, .18, .82)
            long_term_debt = (liabilities-current_liabilities) * r.uniform(.55, .90)
            cash_share = r.uniform(.13, .34) * (1 - 1.3*stress*(disturbance=='cash_crunch'))
            cash = current_assets * max(.02, cash_share)
            inventory = current_assets * r.uniform(.15, .37)
            receivables = current_assets * r.uniform(.13, .32)
            available = .91 * current_assets
            if cash+inventory+receivables > available:
                f = available/(cash+inventory+receivables)
                cash *= f; inventory *= f; receivables *= f
            seasonal = [0.87, .98, 1.02, 1.13][q % 4]
            sales_stress = stress*(1.1 if disturbance == 'sales_fall' else .38)
            revenue = max(8.0, assets * turnover/4 * seasonal * (1-sales_stress)
                          * np.exp(r.normal(0, flow_volatility)))
            margin = base_margin - stress*(.9 if disturbance == 'decline_margin' else .36)
            margin += r.normal(0, .024)
            cogs = revenue * np.clip(1-margin, .42, 1.25)
            gross_profit = revenue-cogs
            op_cost = revenue * max(.02, base_opex + r.normal(0, .016) + stress*.06)
            ebit = gross_profit - op_cost
            depreciation = assets * r.uniform(.002, .010)
            ebitda = ebit+depreciation
            interest = liabilities * r.uniform(.004, .014)
            pretax = ebit-interest
            net = pretax - max(0.0, pretax) * r.uniform(.15, .28)
            working_capital = inventory+receivables-current_liabilities
            change_wc = 0 if prev_working_capital is None else (working_capital-prev_working_capital)
            ocf = net+depreciation - .30*change_wc + r.normal(0, revenue*.028)
            if disturbance == 'cash_crunch':
                ocf -= stress * revenue*.22
            prev_working_capital = working_capital
            d = dict(total_assets=assets, total_liabilities=liabilities, revenue=revenue,
                     net_income=net, current_assets=current_assets,
                     current_liabilities=current_liabilities, cash=cash,
                     long_term_debt=long_term_debt, inventory=inventory,
                     accounts_receivable=receivables, cogs=cogs,
                     gross_profit=gross_profit, ebit=ebit, ebitda=ebitda,
                     operating_cash_flow=ocf)
            company[q] = [d[key] for key in KEYS]
        rows.append(company)
        labels.append('有风险' if risk else '无风险')
    data = np.stack(rows)
    assert data.shape == (80, 20, 15)
    assert sum(v == '有风险' for v in labels) == 16
    return np.arange(1, 81), data, np.asarray(labels)
