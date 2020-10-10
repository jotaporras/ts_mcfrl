import logging
import os
from typing import List


from locations import Order
from network import Arc
import pandas as pd

from network.PhysicalNetwork import PhysicalNetwork


def generate_movement_detail_report(all_movements_history,big_m_cost) -> pd.DataFrame:
    """
    Generates the detailed report of all movements in an episode.
    Movement types are introduced here: Transportation,  Inventory or Delivery.

    :type all_movements_history: List[List[(Arc, int)]]: DataFrame with all movements of the episode.
    big_m_cost: cost to attribute to big m arcs.
    """
    records = []
    unique_keys = set()
    for day_movements in all_movements_history:
        for arc, flow in day_movements:
            movement_type = "N/A"
            transportation_units = 0
            transportation_cost = 0
            inventory_units = 0
            inventory_cost = 0
            customer_units = 0
            customer_cost = 0

            if arc.transportation_arc():
                movement_type = 'Transportation'
                transportation_units = flow
                transportation_cost = arc.cost * flow
            elif arc.inventory_arc():
                movement_type = 'Inventory'
                inventory_units = flow
                inventory_cost = arc.cost * flow
            elif arc.to_customer_arc():
                movement_type = "Delivery"
                customer_units = flow
                customer_cost = arc.cost * flow

            is_big_m =  arc.cost >= big_m_cost
            commodity = arc.commodity


            key = (arc.tail.location.name,
                    arc .head.location.name,
                    arc.tail.time,
                    arc.head.time,
                    commodity)

            if key in unique_keys and movement_type=='Delivery':
                logging.error("This should never happen! The key {key} is added twice in the movement report")
            else:
                unique_keys.add(key)

            records.append(
                (
                    arc.tail.location.name,
                    arc.head.location.name,
                    arc.tail.time,
                    arc.head.time,
                    commodity,
                    arc.tail.kind,
                    arc.head.kind,
                    movement_type,
                    transportation_units,
                    transportation_cost,
                    inventory_units,
                    inventory_cost,
                    customer_units,
                    customer_cost,
                    is_big_m
                )
            )
    movement_detail_report = pd.DataFrame(records, columns=["source_name",
                                                            "destination_name",
                                                            "source_time",
                                                            "destination_time",
                                                            "commodity",
                                                            "source_kind",
                                                            "destination_kind",
                                                            "movement_type",
                                                            "transportation_units",
                                                            "transportation_cost",
                                                            "inventory_units",
                                                            "inventory_cost",
                                                            "customer_units",
                                                            "customer_cost",
                                                            "is_big_m"
                                                            ])
    return movement_detail_report


# Aggregate by source time
def generate_summary_movement_report(movement_detail_report: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by source time for period statistics.
    """
    summary_movement_report = movement_detail_report.groupby('source_time')[[
        "transportation_units",
        "transportation_cost",
        "inventory_units",
        "inventory_cost",
        "customer_units",
        "customer_cost"
    ]].sum()
    summary_movement_report['total_cost'] = summary_movement_report.transportation_cost + summary_movement_report.inventory_cost + summary_movement_report.customer_cost
    return summary_movement_report


def write_experiment_reports(info_object, experiment_name: object,base='data/results'):
    """
    Writes the results of the experiment in CSV.
     info = {
                    'final_orders': final_ords, # The orders, with their final shipping point destinations.
                    'total_costs': self.total_costs, # Total costs per stepwise optimization
                    'approximate_transport_movement_list': self.total_costs, # Total costs per stepwise optimization
                    'approximate_to_customer_cost': approximate_to_customer_cost, # Approximate cost of shipping to customers: total demand multiplied by the default customer transport cost. If the cost is different, this is worthless.
                    'movement_detail_report': movement_detail_report, # DataFrame with all the movements that were made.
                    'summary_movement_report': summary_movement_report  # DataFrame with movements summary per day.
                }
    :rtype: object
    """
    if not os.path.exists(base):
        os.makedirs(base)
    if not os.path.exists(f"{base}/{experiment_name}"):
        os.makedirs(f'{base}/{experiment_name}')
    info_object['movement_detail_report'].to_csv(f"{base}/{experiment_name}/movement_detail_report.csv",index=False)
    info_object['summary_movement_report'].to_csv(f"{base}/{experiment_name}/summary_movement_report.csv",index=False)

def write_single_df_experiment_reports(info_object_list,experiment_name,base='data/results'):
    movement_detail_reports = []
    summary_movement_reports = []
    for i, info in enumerate(info_object_list):
        movement_detail_report = info['movement_detail_report']
        movement_detail_report['episode'] = i
        movement_detail_reports.append(movement_detail_report)
        summary_movement_report = info['summary_movement_report']
        summary_movement_report['episode'] = i
        summary_movement_reports.append(summary_movement_report)
    if len(movement_detail_reports)>0:
        all_movement_detail_reports = pd.concat(movement_detail_reports,axis=0)
        all_summary_movement_reports = pd.concat(summary_movement_reports, axis=0)
        all_experiment_reports_info = {
            'movement_detail_report': all_movement_detail_reports,
            'summary_movement_report': all_summary_movement_reports
        }

        write_experiment_reports(all_experiment_reports_info, experiment_name, base)
    else:
        logging.info("No episodes completed, information skipped.")

def generate_physical_network_valid_dcs(network: PhysicalNetwork):
    rows = []
    for ci,customer in enumerate(network.customers):
        customer_dcs = []
        for dci,dc in enumerate(network.dcs):
            if network.dcs_per_customer_array[ci][dci] == 1:
                rows.append( (customer.name,dc.name) )
    valid_dcs = pd.DataFrame(rows,columns=['customer','valid_dc'])
    return valid_dcs

def write_generate_physical_network_valid_dcs(network: PhysicalNetwork, experiment_name, base="data/results"):
    valid_dcs = generate_physical_network_valid_dcs(network)
    valid_dcs.to_csv(f"{base}/{experiment_name}/valid_dcs.csv",index=False)

