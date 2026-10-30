"""Canonical quarterly financial feature order and Excel column layout."""
FEATURES = [
    ('total_assets', '总资产'), ('total_liabilities', '总负债'),
    ('revenue', '营业收入'), ('net_income', '净利润'),
    ('current_assets', '流动资产'), ('current_liabilities', '流动负债'),
    ('cash', '货币资金'), ('long_term_debt', '长期债务'),
    ('inventory', '存货'), ('accounts_receivable', '应收账款'),
    ('cogs', '营业成本'), ('gross_profit', '毛利润'),
    ('ebit', '息税前利润EBIT'), ('ebitda', '息税折旧摊销前利润EBITDA'),
    ('operating_cash_flow', '经营活动现金流量'),
]
KEYS = [key for key, _ in FEATURES]
NAMES = [name for _, name in FEATURES]
LABELS = ['无风险', '有风险']
N_QUARTERS = 20
N_FEATURES = len(FEATURES)
EXCEL_COLUMNS = ['企业编号', '年份'] + [f'Q{q:02d}_{name}' for q in range(1, 5) for name in NAMES] + ['风险标签']
N_YEARS = 5
QUARTERS_PER_YEAR = 4


def validate_columns(columns):
    if list(columns) != EXCEL_COLUMNS:
        raise ValueError(f'Excel必须为63列，且列顺序严格与schema一致。当前列数={len(columns)}；请不要手工重命名列。')
