from pyomo.environ import ConcreteModel, AbstractModel
from pyomo.environ import Set,Param,Var,Objective,Constraint
from pyomo.environ import PositiveIntegers, NonNegativeReals, Reals
from pyomo.environ import SolverFactory, minimize
from pyomo.environ import value
from pyomo.core.base.param import SimpleParam
import numpy as np


def solve_model(model_instance, solver, tee = True, keepfiles=False):
    if 'path' in solver:
        optimizer = SolverFactory(solver['name'], executable=solver['path'])
    else:
        optimizer = SolverFactory(solver['name'])

    optimizer.solve(model_instance, tee=tee, keepfiles=keepfiles)


    return model_instance


def model_invest(model_data):

    # Create model
    model = ConcreteModel()

    ## SETS
    model.T = Set(dimen=1, ordered=True, initialize=model_data[None]['T']) # Periods
    model.M = Set(dimen=1, ordered=True, initialize=np.array(list(set(model_data[None]['month_order'].values())))) # Months


    ## PARAMETERS
    model.demand                        = Param(model.T, initialize=model_data[None]['demand'])
    model.generation                    = Param(model.T, initialize=model_data[None]['generation'])
    model.generation_pu                 = Param(model.T, initialize=model_data[None]['generation_pu'])
    model.heat_demand                   = Param(model.T, initialize=model_data[None]['heat_demand'])

    model.battery_soc_min               = Param(initialize=model_data[None]['battery_soc_min'])
    model.battery_capacity              = Param(initialize=model_data[None]['battery_capacity'])
    model.battery_charge_max            = Param(initialize=model_data[None]['battery_charge_max'])
    model.battery_discharge_max         = Param(initialize=model_data[None]['battery_discharge_max'])
    model.battery_efficiency_charge     = Param(initialize=model_data[None]['battery_efficiency_charge'])
    model.battery_efficiency_discharge  = Param(initialize=model_data[None]['battery_efficiency_discharge'])
    model.battery_soc_ini               = Param(initialize=model_data[None]['battery_soc_ini'])
    model.battery_soc_fin               = Param(initialize=model_data[None]['battery_soc_fin'])
    model.battery_grid_charging         = Param(initialize=model_data[None]['battery_grid_charging'])
    
    model.boiler_efficiency             = Param(initialize=model_data[None]['boiler_efficiency'])
    model.heat_pump_cop                 = Param(initialize=model_data[None]['heat_pump_cop'])
    model.fuel_price                    = Param(initialize=model_data[None]['fuel_price'])
    
    model.price_spot_buy                = Param(model.T, initialize=model_data[None]['price_spot_buy'])
    model.price_spot_sell               = Param(model.T, initialize=model_data[None]['price_spot_sell'])
    model.price_fcrn                    = Param(model.T, initialize=model_data[None]['price_fcrn'])
    model.price_fcrd_up                 = Param(model.T, initialize=model_data[None]['price_fcrd_up'])
    model.price_fcrd_down               = Param(model.T, initialize=model_data[None]['price_fcrd_down'])
    model.price_balancing_up            = Param(model.T, initialize=model_data[None]['price_balancing_up'])
    model.price_balancing_down          = Param(model.T, initialize=model_data[None]['price_balancing_down'])
    
    model.volume_fcrn_up                = Param(model.T, initialize=model_data[None]['volume_fcrn_up'])   
    model.volume_fcrn_down              = Param(model.T, initialize=model_data[None]['volume_fcrn_down'])   
    model.volume_fcrd_up                = Param(model.T, initialize=model_data[None]['volume_fcrd_up'])
    model.volume_fcrd_down              = Param(model.T, initialize=model_data[None]['volume_fcrd_down'])
    
    model.grid_fee_fixed                = Param(initialize=model_data[None]['grid_fee_fixed'])
    model.grid_fee_energy_import        = Param(model.T, within=Reals, initialize=model_data[None]['grid_fee_energy_import'])
    model.grid_fee_energy_export        = Param(model.T, within=Reals, initialize=model_data[None]['grid_fee_energy_export'])
    model.grid_fee_power_import         = Param(model.T, within=Reals, initialize=model_data[None]['grid_fee_power_import'])
    model.grid_fee_power_export         = Param(model.T, within=Reals, initialize=model_data[None]['grid_fee_power_export'])

    model.invest_cost_pu_pv             = Param(initialize=model_data[None]['invest_cost_pu_pv'])    
    model.invest_capacity_min_pv        = Param(initialize=model_data[None]['invest_capacity_min_pv'])
    model.invest_capacity_max_pv        = Param(initialize=model_data[None]['invest_capacity_max_pv'])
    model.invest_cost_pu_battery        = Param(initialize=model_data[None]['invest_cost_pu_battery'])
    model.invest_capacity_min_battery   = Param(initialize=model_data[None]['invest_capacity_min_battery'])
    model.invest_capacity_max_battery   = Param(initialize=model_data[None]['invest_capacity_max_battery'])
    model.invest_cost_pu_hp             = Param(initialize=model_data[None]['invest_cost_pu_hp']) 
    model.invest_capacity_max_hp        = Param(initialize=model_data[None]['invest_capacity_max_hp'])
    model.invest_cost_pu_boiler         = Param(initialize=model_data[None]['invest_cost_pu_boiler'])
    model.invest_capacity_max_boiler    = Param(initialize=model_data[None]['invest_capacity_max_boiler'])
    
    model.dt                            = Param(initialize=model_data[None]['dt'])


    ## VARIABLES
    model.COST_INV                      = Var(within=Reals)
    model.COST_INV_HEAT                 = Var(within=Reals)
    model.COST_ENERGY_SPOT              = Var(model.T, within=Reals)
    model.COST_GRID_ENERGY_IMPORT       = Var(model.T, within=NonNegativeReals)
    model.COST_GRID_ENERGY_EXPORT       = Var(model.T, within=NonNegativeReals)
    model.COST_GRID_POWER_IMPORT        = Var(model.T, within=NonNegativeReals)
    model.COST_GRID_POWER_EXPORT        = Var(model.T, within=NonNegativeReals)
    model.COST_GRID_POWER               = Var(model.T, within=Reals)
    model.COST_GRID_POWER_MAX           = Var(model.M, within=Reals)
    model.COST_GRID_FIXED               = Var(within=Reals)
    model.REVENUE_CAPACITY_FCRN         = Var(model.T, within=Reals)
    model.REVENUE_CAPACITY_FCRD         = Var(model.T, within=Reals)
    model.REVENUE_ENERGY_FCRN           = Var(model.T, within=Reals)
    model.REVENUE_ENERGY_FCRD           = Var(model.T, within=Reals)    
    
    model.P_TOTAL_BUY                   = Var(model.T, within=NonNegativeReals)
    model.P_TOTAL_SELL                  = Var(model.T, within=NonNegativeReals)
    model.P_SPOT_BUY                    = Var(model.T, within=NonNegativeReals)
    model.P_SPOT_SELL                   = Var(model.T, within=NonNegativeReals)
    model.P_FCRN_BUY                    = Var(model.T, within=NonNegativeReals)
    model.P_FCRN_SELL                   = Var(model.T, within=NonNegativeReals)
    model.P_FCRD_BUY                    = Var(model.T, within=NonNegativeReals)
    model.P_FCRD_SELL                   = Var(model.T, within=NonNegativeReals)
    
    model.P_FCRN                        = Var(model.T, within=NonNegativeReals)
    model.P_FCRD_DOWN                   = Var(model.T, within=NonNegativeReals)
    model.P_FCRD_UP                     = Var(model.T, within=NonNegativeReals)
    
    model.B_SOC                         = Var(model.T, within=NonNegativeReals)
    model.B_IN                          = Var(model.T, within=NonNegativeReals)
    model.B_OUT                         = Var(model.T, within=NonNegativeReals)

    model.CAP_PV                        = Var(within=NonNegativeReals, bounds=(model.invest_capacity_min_pv, model.invest_capacity_max_pv))
    model.CAP_BAT                       = Var(within=NonNegativeReals, bounds=(model.invest_capacity_min_battery, model.invest_capacity_max_battery))
    model.CAP_BO                        = Var(within=NonNegativeReals)
    model.CAP_HP                        = Var(within=NonNegativeReals)
    
    model.GEN                           = Var(model.T, within=Reals)
    
    model.Q_HP                          = Var(model.T, within=NonNegativeReals)
    model.Q_BO                          = Var(model.T, within=NonNegativeReals)
    model.P_HP                          = Var(model.T, within=NonNegativeReals)
    model.F_BO                          = Var(model.T, within=NonNegativeReals)
    model.COST_FUEL                     = Var(model.T, within=Reals)


    ## OBJECTIVE FUNCTION
    # Minimize cost
    def total_cost(model):
        return model.COST_INV + model.COST_INV_HEAT \
        + sum(model.COST_ENERGY_SPOT[t] for t in model.T) \
        - sum(model.REVENUE_ENERGY_FCRN[t] for t in model.T) \
        - sum(model.REVENUE_CAPACITY_FCRN[t] for t in model.T) \
        - sum(model.REVENUE_ENERGY_FCRD[t] for t in model.T) \
        - sum(model.REVENUE_CAPACITY_FCRD[t] for t in model.T) \
        + model.COST_GRID_FIXED \
        + sum(model.COST_GRID_ENERGY_IMPORT[t] for t in model.T) \
        + sum(model.COST_GRID_ENERGY_EXPORT[t] for t in model.T) \
        + sum(model.COST_GRID_POWER_MAX[m] for m in model.M) \
        + sum(model.COST_FUEL[t] for t in model.T)
    model.total_cost = Objective(rule=total_cost, sense=minimize)



    ## CONSTRAINTS
    
    ########################### INVESTMENT COSTS ###########################
    # Investment cost PV and battery
    def investment_cost(model):
        return model.COST_INV == model.invest_cost_pu_pv*model.CAP_PV + model.invest_cost_pu_battery*model.CAP_BAT
    model.investment_cost = Constraint(rule=investment_cost)
    
    # Investment cost gas boiler and heat pump
    def investment_cost_heating(model):
        return model.COST_INV_HEAT == model.invest_cost_pu_hp*model.CAP_HP + model.invest_cost_pu_boiler*model.CAP_BO
    model.investment_cost_heating = Constraint(rule=investment_cost_heating)
    
    
    
    ########################### DAYAHEAD MARKET ###########################
    # Energy cost
    def energy_cost_spot(model, t):
        return model.COST_ENERGY_SPOT[t] == model.price_spot_buy[t]*model.P_SPOT_BUY[t]*model.dt - model.price_spot_sell[t]*model.P_SPOT_SELL[t]*model.dt
    model.energy_cost_spot = Constraint(model.T, rule=energy_cost_spot)
    


    ########################### FCRN ###########################    
    # Energy revenue
    def energy_revenue_fcrn(model, t):
        return model.REVENUE_ENERGY_FCRN[t] == model.price_balancing_up[t]*model.P_FCRN_SELL[t]*model.dt - model.price_balancing_down[t]*model.P_FCRN_BUY[t]*model.dt
    model.energy_revenue_fcrn = Constraint(model.T, rule=energy_revenue_fcrn)
    
    # Reserve capacity revenue
    def capacity_revenue_fcrn(model, t):
        return model.REVENUE_CAPACITY_FCRN[t] == model.price_fcrn[t]*model.dt*model.P_FCRN[t]
    model.capacity_revenue_fcrn = Constraint(model.T, rule=capacity_revenue_fcrn)
    
    # Activated down regulation volume
    def volume_fcrn_buy(model, t):
        return model.P_FCRN_BUY[t] == model.volume_fcrn_down[t]*model.P_FCRN[t]
    model.volume_fcrn_buy = Constraint(model.T, rule=volume_fcrn_buy)
    
    # Activated up regulation volume
    def volume_fcrn_sell(model, t):
        return model.P_FCRN_SELL[t] == model.volume_fcrn_up[t]*model.P_FCRN[t]
    model.volume_fcrn_sell = Constraint(model.T, rule=volume_fcrn_sell)



    ########################### FCRD ###########################    
    # Energy revenue
    def energy_revenue_fcrd(model, t):
        return model.REVENUE_ENERGY_FCRD[t] == model.price_balancing_up[t]*model.P_FCRD_SELL[t]*model.dt - model.price_balancing_down[t]*model.P_FCRD_BUY[t]*model.dt
    model.energy_revenue_fcrd = Constraint(model.T, rule=energy_revenue_fcrd)
    
    # Reserve capacity revenue
    def capacity_revenue_fcrd(model, t):
        return model.REVENUE_CAPACITY_FCRD[t] == model.price_fcrd_up[t]*model.dt*model.P_FCRD_UP[t] + model.price_fcrd_down[t]*model.dt*model.P_FCRD_DOWN[t]
    model.capacity_revenue_fcrd = Constraint(model.T, rule=capacity_revenue_fcrd)
    
    # Activated down regulation volume
    def volume_fcrd_buy(model, t):
        return model.P_FCRD_BUY[t] == model.volume_fcrd_down[t]*model.P_FCRD_DOWN[t]
    model.volume_fcrd_buy = Constraint(model.T, rule=volume_fcrd_buy)
    
    # Activated up regulation volume
    def volume_fcrd_sell(model, t):
        return model.P_FCRD_SELL[t] == model.volume_fcrd_up[t]*model.P_FCRD_UP[t]
    model.volume_fcrd_sell = Constraint(model.T, rule=volume_fcrd_sell)



    ########################### GRID FEES ###########################
    # Grid fixed cost
    def grid_fixed_cost(model):
        return model.COST_GRID_FIXED == model.grid_fee_fixed*len(model.M)
    model.grid_fixed_cost = Constraint(rule=grid_fixed_cost)
    
    # Grid energy import cost
    def grid_energy_import_cost(model, t):
        return model.COST_GRID_ENERGY_IMPORT[t] >= model.grid_fee_energy_import[t]*(model.P_TOTAL_BUY[t]-model.P_TOTAL_SELL[t])*model.dt
    model.grid_energy_import_cost = Constraint(model.T, rule=grid_energy_import_cost)
    
    # Grid energy export cost
    def grid_energy_export_cost(model, t):
        return model.COST_GRID_ENERGY_EXPORT[t] >= model.grid_fee_energy_export[t]*(model.P_TOTAL_SELL[t]-model.P_TOTAL_BUY[t])*model.dt
    model.grid_energy_export_cost = Constraint(model.T, rule=grid_energy_export_cost)

    # Grid power import cost
    def grid_power_import_cost(model, t):
        return model.COST_GRID_POWER_IMPORT[t] >= model.grid_fee_power_import[t]*(model.P_TOTAL_BUY[t]-model.P_TOTAL_SELL[t])
    model.grid_power_import_cost = Constraint(model.T, rule=grid_power_import_cost)
    
    # Grid power export cost
    def grid_power_export_cost(model, t):
        return model.COST_GRID_POWER_EXPORT[t] >= model.grid_fee_power_export[t]*(model.P_TOTAL_SELL[t]-model.P_TOTAL_BUY[t])
    model.grid_power_export_cost = Constraint(model.T, rule=grid_power_export_cost)

    # Grid power total cost
    def grid_power_cost(model, t):
        return model.COST_GRID_POWER[t] == model.COST_GRID_POWER_IMPORT[t] + model.COST_GRID_POWER_EXPORT[t]
    model.grid_power_cost = Constraint(model.T, rule=grid_power_cost)

    # Max grid cost
    def max_grid_power_cost(model, t):
        return model.COST_GRID_POWER_MAX[model_data[None]['month_order'][t]] >= model.COST_GRID_POWER[t]
    model.max_grid_power_cost = Constraint(model.T, rule=max_grid_power_cost)



    ########################### BALANCE EQUATIONS ###########################
    # Power sell
    def power_sell_total(model, t):
        return model.P_TOTAL_SELL[t] == model.P_SPOT_SELL[t] + model.P_FCRN_SELL[t] + model.P_FCRD_SELL[t]
    model.power_sell_total = Constraint(model.T, rule=power_sell_total)
    
    # Power sell limit
    def power_sell_limit(model, t):
        return model.P_TOTAL_SELL[t] <= model.GEN[t] + model.B_OUT[t]
    model.power_sell_limit = Constraint(model.T, rule=power_sell_limit)
    
    # Power buy
    def power_buy_total(model, t):
        return model.P_TOTAL_BUY[t] == model.P_SPOT_BUY[t] + model.P_FCRN_BUY[t] + model.P_FCRD_BUY[t]
    model.power_buy_total = Constraint(model.T, rule=power_buy_total)

    # Power generation
    def power_generation(model, t):
        return model.GEN[t] == model.generation[t] + model.generation_pu[t]*model.CAP_PV
    model.power_generation = Constraint(model.T, rule=power_generation)
    
    # Energy balance
    def power_balance(model, t):
        return model.P_TOTAL_SELL[t] - model.P_TOTAL_BUY[t] == model.GEN[t] + model.B_OUT[t] - model.B_IN[t] - model.P_HP[t] - model.demand[t]
    model.power_balance = Constraint(model.T, rule=power_balance)

    # Battery charging from grid
    def no_grid_charging(model, t):
        if value(model.battery_grid_charging) == False:
            return model.P_TOTAL_BUY[t] <= model.demand[t]   
        else:
            return model.P_TOTAL_BUY[t] <= model.demand[t] + model.B_IN[t]
    model.no_grid_charging = Constraint(model.T, rule=no_grid_charging)

    # Battery energy balance
    def battery_soc(model, t):
        if t==model.T.first():
            return model.B_SOC[t] - model.battery_soc_ini*model.CAP_BAT == model.battery_efficiency_charge*model.B_IN[t]*model.dt  - (1/model.battery_efficiency_discharge)*model.B_OUT[t]*model.dt
        else:
            return model.B_SOC[t] - model.B_SOC[model.T.prev(t)] == model.battery_efficiency_charge*model.B_IN[t]*model.dt  - (1/model.battery_efficiency_discharge)*model.B_OUT[t]*model.dt
    model.battery_soc = Constraint(model.T, rule=battery_soc)

    # Battery SOC upper limit
    def battery_soc_upper_limit(model, t):
        return model.B_SOC[t] <= model.CAP_BAT
    model.battery_soc_upper_limit = Constraint(model.T, rule=battery_soc_upper_limit)

    # Battery SOC lower limit
    def battery_soc_lower_limit(model, t):
        return model.B_SOC[t] >= model.battery_soc_min*model.CAP_BAT
    model.battery_soc_lower_limit = Constraint(model.T, rule=battery_soc_lower_limit)


    # Reserve capacity up regulation limit
    def reserve_up_limit(model, t):
        if t==model.T.first():
            return model.P_FCRN[t] + model.P_FCRD_UP[t] <= (model.battery_soc_ini*model.CAP_BAT - model.battery_soc_min*model.CAP_BAT)/model.dt
        else:
            return model.P_FCRN[t] + model.P_FCRD_UP[t] <= (model.B_SOC[model.T.prev(t)] - model.battery_soc_min*model.CAP_BAT)/model.dt
    model.reserve_up_limit = Constraint(model.T, rule=reserve_up_limit)
    
    
    # Reserve capacity down regulation limit
    def reserve_down_limit(model, t):
        if t==model.T.first():
            return model.P_FCRN[t] + model.P_FCRD_DOWN[t] <= (model.CAP_BAT - model.battery_soc_ini*model.CAP_BAT)/model.dt
        else:
            return model.P_FCRN[t] + model.P_FCRD_DOWN[t] <= (model.CAP_BAT - model.B_SOC[model.T.prev(t)])/model.dt
    model.reserve_down_limit = Constraint(model.T, rule=reserve_down_limit)


    # Fix battery soc in the last period
    if value(model.battery_soc_fin) > 0:
        model.B_SOC[model.T.last()].fix(model.battery_soc_fin*model.CAP_BAT)


    ########################### HEAT EQUATIONS ###########################
    # Heat balance
    def heat_balance(model, t):
        return model.Q_BO[t] + model.Q_HP[t] == model.heat_demand[t]
    model.heat_balance = Constraint(model.T, rule=heat_balance)

    # Boiler heat generation
    def fuel_boiler_gen(model, t):
        return model.F_BO[t] == (1/model.boiler_efficiency)*model.Q_BO[t]
    model.fuel_boiler_gen = Constraint(model.T, rule=fuel_boiler_gen)
    
    # Boiler heat generation cap
    def boiler_gen_cap(model, t):
        return model.Q_BO[t] <= model.CAP_BO
    model.boiler_gen_cap = Constraint(model.T, rule=boiler_gen_cap)
    
    # Fuel consumption cost
    def boiler_fuel_cost(model, t):
        return model.COST_FUEL[t] == model.fuel_price*model.F_BO[t]*model.dt
    model.boiler_fuel_cost = Constraint(model.T, rule=boiler_fuel_cost)

    # Heat pump heat generation
    def heat_pump_gen(model, t):
        return model.Q_HP[t] == model.heat_pump_cop*model.P_HP[t]
    model.heat_pump_gen = Constraint(model.T, rule=heat_pump_gen)
    
    # Heat pump heat generation cap
    def heat_pump_gen_cap(model, t):
        return model.Q_HP[t] <= model.CAP_HP
    model.heat_pump_gen_cap = Constraint(model.T, rule=heat_pump_gen_cap)
    
    # Heat pump heat generation cap
    def heat_pump_capacity_cap(model):
        return model.CAP_HP <= model.invest_capacity_max_hp
    model.heat_pump_capacity_cap = Constraint(rule=heat_pump_capacity_cap)
    
    # Heat pump heat generation cap
    def boiler_capacity_cap(model):
        return model.CAP_BO <= model.invest_capacity_max_boiler
    model.boiler_capacity_cap = Constraint(rule=boiler_capacity_cap)
    
    return model


def model_invest_input(data):
    
    
    def annuity_rate(discount_rate, salvage_rate, life_span):
        
        anrate = (1-salvage_rate)*discount_rate*(1+discount_rate)**life_span/((1+discount_rate)**life_span-1)
        return anrate


    periods = np.arange(1, len(data['generation_pu'])+1) if 'generation_pu' in data else np.arange(1, len(data['demand'])+1)
    
    generation_pu = dict(zip(periods,  data['generation_pu'])) if 'generation_pu' in data else dict(zip(periods,  [0] * len(periods)))
    demand = dict(zip(periods,  data['demand'])) if 'demand' in data else dict(zip(periods,  [0] * len(periods)))
    generation = dict(zip(periods,  data['generation'])) if 'generation' in data else dict(zip(periods,  [0] * len(periods)))
    heat_demand = dict(zip(periods,  data['heat_demand'])) if 'heat_demand' in data else dict(zip(periods,  [0] * len(periods)))
    
    pv_lifespan = data['pv_lifespan'] if 'pv_lifespan' in data else 25.0
    pv_salvage_rate = data['pv_salvage_rate'] if 'pv_salvage_rate' in data else 0.0
    
    hp_lifespan = data['hp_lifespan'] if 'hp_lifespan' in data else 25.0
    hp_salvage_rate = data['hp_salvage_rate'] if 'hp_salvage_rate' in data else 0.0
    boiler_lifespan = data['boiler_lifespan'] if 'boiler_lifespan' in data else 25.0
    boiler_salvage_rate = data['boiler_salvage_rate'] if 'boiler_salvage_rate' in data else 0.0

    battery_capacity = data['battery_capacity'] if 'battery_capacity' in data else 0.0
    battery_soc_min = data['battery_soc_min'] if 'battery_soc_min' in data else 0.0
    battery_charge_max = data['battery_charge_max'] if 'battery_charge_max' in data else 1.0
    battery_discharge_max = data['battery_discharge_max'] if 'battery_discharge_max' in data else 1.0
    battery_efficiency_charge = data['battery_efficiency_charge'] if 'battery_efficiency_charge' in data else 0.95
    battery_efficiency_discharge = data['battery_efficiency_discharge'] if 'battery_efficiency_discharge' in data else 0.95
    battery_soc_ini = data['battery_soc_ini'] if 'battery_soc_ini' in data else 0.0
    battery_soc_fin = data['battery_soc_fin'] if 'battery_soc_fin' in data else 0.0 
    battery_grid_charging = data['battery_grid_charging'] if 'battery_grid_charging' in data else True
    battery_lifespan = data['battery_lifespan'] if 'battery_lifespan' in data else 10.0
    battery_salvage_rate = data['battery_salvage_rate'] if 'battery_salvage_rate' in data else 0.0
    
    price_spot_buy = dict(zip(periods,  data['price_spot_buy'])) if 'price_spot_buy' in data else dict(zip(periods,  [0] * len(periods)))
    price_spot_sell = dict(zip(periods,  data['price_spot_sell'])) if 'price_spot_sell' in data else dict(zip(periods,  [0] * len(periods)))
    price_fcrn = dict(zip(periods,  data['price_fcrn'])) if 'price_fcrn' in data else dict(zip(periods,  [0] * len(periods)))    
    price_fcrd_up = dict(zip(periods,  data['price_fcrd_up'])) if 'price_fcrd_up' in data else dict(zip(periods,  [0] * len(periods)))
    price_fcrd_down = dict(zip(periods,  data['price_fcrd_down'])) if 'price_fcrd_down' in data else dict(zip(periods,  [0] * len(periods)))
    price_balancing_up = dict(zip(periods,  data['price_balancing_up'])) if 'price_balancing_up' in data else price_spot_buy
    price_balancing_down = dict(zip(periods,  data['price_balancing_down'])) if 'price_balancing_down' in data else price_spot_buy
    
    volume_fcrn_up = dict(zip(periods,  data['volume_fcrn_up'])) if 'volume_fcrn_up' in data else dict(zip(periods,  [0] * len(periods)))
    volume_fcrn_down = dict(zip(periods,  data['volume_fcrn_down'])) if 'volume_fcrn_down' in data else dict(zip(periods,  [0] * len(periods)))
    volume_fcrd_up = dict(zip(periods,  data['volume_fcrd_up'])) if 'volume_fcrd_up' in data else dict(zip(periods,  [0] * len(periods)))
    volume_fcrd_down = dict(zip(periods,  data['volume_fcrd_down'])) if 'volume_fcrd_down' in data else dict(zip(periods,  [0] * len(periods)))
    
    grid_fee_fixed = data['grid_fee_fixed'] if 'grid_fee_fixed' in data else 0.0
    grid_fee_energy_import = dict(zip(periods,  data['grid_fee_energy_import'])) if 'grid_fee_energy_import' in data else dict(zip(periods,  [0] * len(periods)))
    grid_fee_energy_export = dict(zip(periods,  data['grid_fee_energy_export'])) if 'grid_fee_energy_export' in data else dict(zip(periods,  [0] * len(periods)))
    grid_fee_power_import = dict(zip(periods,  data['grid_fee_power_import'])) if 'grid_fee_power_import' in data else dict(zip(periods,  [0] * len(periods)))
    grid_fee_power_export = dict(zip(periods,  data['grid_fee_power_export'])) if 'grid_fee_power_export' in data else dict(zip(periods,  [0] * len(periods)))

    discount_rate = data['discount_rate'] if 'discount_rate' in data else 0.0

    annuity_rate_pv = annuity_rate(discount_rate, pv_salvage_rate, pv_lifespan)
    annuity_rate_battery = annuity_rate(discount_rate, battery_salvage_rate, battery_lifespan)
    annuity_rate_hp = annuity_rate(discount_rate, hp_salvage_rate, hp_lifespan)
    annuity_rate_boiler = annuity_rate(discount_rate, boiler_salvage_rate, boiler_lifespan)

    invest_cost_pu_battery = data['invest_cost_pu_battery']*annuity_rate_battery if 'invest_cost_pu_battery' in data else 0.0
    invest_cost_pu_pv = data['invest_cost_pu_pv']*annuity_rate_pv if 'invest_cost_pu_pv' in data else 0.0
    invest_capacity_min_pv = data['invest_capacity_min_pv'] if 'invest_capacity_min_pv' in data else 0.0
    invest_capacity_max_pv = data['invest_capacity_max_pv'] if 'invest_capacity_max_pv' in data else 0.0
    invest_capacity_min_battery = data['invest_capacity_min_battery'] if 'invest_capacity_min_battery' in data else 0.0
    invest_capacity_max_battery = data['invest_capacity_max_battery'] if 'invest_capacity_max_battery' in data else 0.0
    invest_cost_pu_hp = data['invest_cost_pu_hp']*annuity_rate_hp if 'invest_cost_pu_hp' in data else 0.0
    invest_capacity_max_hp = data['invest_capacity_max_hp'] if 'invest_capacity_max_hp' in data else 0.0
    invest_cost_pu_boiler = data['invest_cost_pu_boiler']*annuity_rate_boiler if 'invest_cost_pu_boiler' in data else 0.0
    invest_capacity_max_boiler = data['invest_capacity_max_boiler'] if 'invest_capacity_max_boiler' in data else 0.0
    
    
    fuel_price = data['fuel_price'] if 'fuel_price' in data else 0.0
    boiler_efficiency = data['boiler_efficiency'] if 'boiler_efficiency' in data else 1.0
    heat_pump_cop = data['heat_pump_cop'] if 'heat_pump_cop' in data else 1.0
    
    dt = data['dt'] if 'dt' in data else 1.0
    month_order = dict(zip(periods,  data['month_order'])) if 'month_order' in data else dict(zip(periods,  [1] * len(periods)))


    # Create model data input dictionary
    model_data = {None: {
        'T': periods,

        'generation': generation,
        'generation_pu': generation_pu,
        'demand': demand,
        'heat_demand': heat_demand,

        'battery_soc_min': battery_soc_min,
        'battery_capacity': battery_capacity,
        'battery_charge_max': battery_charge_max,
        'battery_discharge_max': battery_discharge_max,
        'battery_efficiency_charge': battery_efficiency_charge,
        'battery_efficiency_discharge': battery_efficiency_discharge,
        'battery_grid_charging': battery_grid_charging,
        'battery_soc_ini': battery_soc_ini,
        'battery_soc_fin': battery_soc_fin,

        'boiler_efficiency': boiler_efficiency,
        'heat_pump_cop': heat_pump_cop,

        'price_spot_buy': price_spot_buy,
        'price_spot_sell': price_spot_sell,
        'price_fcrn': price_fcrn,
        'volume_fcrn_up': volume_fcrn_up,
        'volume_fcrn_down': volume_fcrn_down,
        'price_fcrd_up': price_fcrd_up,
        'price_fcrd_down': price_fcrd_down,
        'volume_fcrd_up': volume_fcrd_up,
        'volume_fcrd_down': volume_fcrd_down,
        'price_balancing_up': price_balancing_up,
        'price_balancing_down': price_balancing_down,
        
        'grid_fee_fixed': grid_fee_fixed,
        'grid_fee_energy_import': grid_fee_energy_import,
        'grid_fee_energy_export': grid_fee_energy_export,
        'grid_fee_power_import': grid_fee_power_import,
        'grid_fee_power_export': grid_fee_power_export,
        
        'invest_cost_pu_battery': invest_cost_pu_battery,
        'invest_capacity_min_battery': invest_capacity_min_battery,
        'invest_capacity_max_battery': invest_capacity_max_battery,
        'invest_cost_pu_pv': invest_cost_pu_pv,
        'invest_capacity_min_pv': invest_capacity_min_pv,
        'invest_capacity_max_pv': invest_capacity_max_pv,
        'invest_cost_pu_hp': invest_cost_pu_hp,
        'invest_capacity_max_hp': invest_capacity_max_hp,
        'invest_cost_pu_boiler': invest_cost_pu_boiler,
        'invest_capacity_max_boiler': invest_capacity_max_boiler,

        'fuel_price': fuel_price,
        
        'month_order': month_order,
        'dt': dt,
    }}

    return model_data


def model_invest_results(solution):
    
    s = dict()
    
    s['cost_total'] = solution.total_cost()
    
    s['investment_cost'] = value(solution.COST_INV)
    s['investment_cost_heating'] = value(solution.COST_INV_HEAT)
    
    s['cost_energy_spot'] = value(solution.COST_ENERGY_SPOT[:])
    s['revenue_energy_fcrn'] = value(solution.REVENUE_ENERGY_FCRN[:])
    s['revenue_capacity_fcrn'] = value(solution.REVENUE_CAPACITY_FCRN[:])
    s['revenue_energy_fcrd'] = value(solution.REVENUE_ENERGY_FCRD[:])
    s['revenue_capacity_fcrd'] = value(solution.REVENUE_CAPACITY_FCRD[:])
    
    s['cost_grid_energy_import'] = value(solution.COST_GRID_ENERGY_IMPORT[:])
    s['cost_grid_energy_export'] = value(solution.COST_GRID_ENERGY_EXPORT[:])
    s['cost_grid_power_import'] = value(solution.COST_GRID_POWER_IMPORT[:])
    s['cost_grid_power_export'] = value(solution.COST_GRID_POWER_EXPORT[:])
    s['cost_grid_power'] = value(solution.COST_GRID_POWER[:])
    s['cost_grid_power_max'] = value(solution.COST_GRID_POWER_MAX[:])
    s['cost_grid_power_fixed'] = value(solution.COST_GRID_FIXED)
    
    s['power_total_buy'] = value(solution.P_TOTAL_BUY[:])
    s['power_total_sell'] = value(solution.P_TOTAL_SELL[:])
    
    s['power_spot_buy'] = value(solution.P_SPOT_BUY[:])
    s['power_spot_sell'] = value(solution.P_SPOT_SELL[:])
    s['power_fcrn'] = value(solution.P_FCRN[:])
    s['power_fcrn_activated_down'] = value(solution.P_FCRN_BUY[:])
    s['power_fcrn_activated_up'] = value(solution.P_FCRN_SELL[:])
    s['power_fcrd_down'] = value(solution.P_FCRD_DOWN[:])
    s['power_fcrd_up'] = value(solution.P_FCRD_UP[:])
    s['power_fcrd_activated_down'] = value(solution.P_FCRD_BUY[:])
    s['power_fcrd_activated_up'] = value(solution.P_FCRD_SELL[:])
    
    s['battery_soc'] = value(solution.B_SOC[:])
    s['battery_charge'] = value(solution.B_IN[:])
    s['battery_discharge'] = value(solution.B_OUT[:])
    
    s['pv_capacity_invest'] = value(solution.CAP_PV)
    s['battery_capacity_invest'] = value(solution.CAP_BAT)
    s['heat_pump_capacity_invest'] = value(solution.CAP_HP)
    s['boiler_capacity_invest'] = value(solution.CAP_BO)

    s['total_generation'] = value(solution.GEN[:])
    s['power_consumption_heat_pump'] = value(solution.P_HP[:])
    s['fuel_consumption_boiler'] = value(solution.F_BO[:])
    s['heat_generation_boiler'] = value(solution.Q_BO[:])
    s['heat_generation_heat_pump'] = value(solution.Q_HP[:])
    
    return s

