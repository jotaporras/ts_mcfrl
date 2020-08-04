import os
from typing import List

from locations import Order
from network import Arc
import pandas as pd


def generate_movement_detail_report(all_movements_history) -> pd.DataFrame:
    """
    Generates the detailed report of all movements in an episode.
    Movement types are introduced here: Transportation,  Inventory or Delivery.

    :type all_movements_history: List[List[(Arc, int)]]: DataFrame with all movements of the episode.
    """
    records = []
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

            records.append(
                (
                    arc.tail.location.name,
                    arc.head.location.name,
                    arc.tail.time,
                    arc.head.time,
                    arc.tail.kind,
                    arc.head.kind,
                    movement_type,
                    transportation_units,
                    transportation_cost,
                    inventory_units,
                    inventory_cost,
                    customer_units,
                    customer_cost
                )
            )
    movement_detail_report = pd.DataFrame(records, columns=["source_name",
                                                            "destination_name",
                                                            "source_time",
                                                            "destination_time",
                                                            "source_kind",
                                                            "destination_kind",
                                                            "movement_type",
                                                            "transportation_units",
                                                            "transportation_cost",
                                                            "inventory_units",
                                                            "inventory_cost",
                                                            "customer_units",
                                                            "customer_cost"
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