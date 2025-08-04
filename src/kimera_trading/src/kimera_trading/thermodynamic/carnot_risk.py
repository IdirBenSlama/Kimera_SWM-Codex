class CarnotCycleRiskModel:
    """
    A risk model inspired by the Carnot cycle.

    Efficiency = 1 - (T_cold / T_hot)
    """

    def __init__(self, hot_temp=300, cold_temp=273):
        self.hot_temp = hot_temp
        self.cold_temp = cold_temp

    def calculate_efficiency(self):
        return 1 - (self.cold_temp / self.hot_temp)
