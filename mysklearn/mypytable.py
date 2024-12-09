"""
Programmer: Jaden Phan
Class: CPSC 322, Fall 2024
Programming Assignment #4
10/18/2024

Description: This program practices making table joins, from PA2
"""

import copy
import csv
from tabulate import tabulate
import random


# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    # TODO: Extract value function, somewhere in notes...

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        ret_col = []

        if isinstance(col_identifier, str):
            if col_identifier in self.column_names:
                col_index = self.column_names.index(col_identifier)
            else:
                raise ValueError(f"Column name '{col_identifier}' not found.")
        elif isinstance(col_identifier, int):
            if 0 <= col_identifier < len(self.column_names):
                col_index = col_identifier
            else:
                raise ValueError(f"Column index '{col_identifier}' out of range.")
        else:
            raise ValueError("col_identifier must be a string or an integer.")

        for row in self.data:
            if include_missing_values or row[col_index] != "NA":
                ret_col.append(row[col_index])

        return ret_col

    def add_column(self, col_name, values):
        """Adds new column to the table

        Args:
            col_name (str): name of new column
            values (list): list of values for new column
        """
        if col_name in self.column_names:
            raise ValueError("This column already exists!")

        if len(values) != len(self.data):
            raise ValueError(
                "The number of values must match the number of rows in the table."
            )

        self.column_names.append(col_name)

        for row, value in zip(self.data, values):
            row.append(value)

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for i in range(len(row)):
                try:
                    row[i] = float(row[i])
                except ValueError:
                    continue

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        self.data = [
            row for i, row in enumerate(self.data) if i not in row_indexes_to_drop
        ]

    def drop_duplicate_seasons(self, year):
        """Prevents data leaks by removing rows from the same season

        Args:
            year (int): year of season we want to drop
        """

        col_index = self.column_names.index("season")

        drop_indices = []

        for i in range(len(self.data)):
            if self.data[i][col_index] == year:
                drop_indices.append(i)

        self.drop_rows(drop_indices)

    def random_down_sample(self, num_rows, rand_seed=1010):
        """Randomly samples rows from dataset, returns a new MyPyTable containing sampled rows.

        Args:
            rand_seed (int): integer that will set random seed
            num_rows (int): number of rows left in down sampled dataset

        Returns:
            MyPyTable: return sorted new MyPyTable containing new dataset with num_rows rows
        """

        # Seed random
        random.seed(rand_seed)

        # Shuffle indices
        sampled_indices = random.sample(range(len(self.data)), num_rows)

        # Extract the rows from sampled_indices
        sampled_data = [self.data[i] for i in sampled_indices]

        # Return a new MyPyTable with downsampled data
        return MyPyTable(self.column_names, sampled_data)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, mode="r", newline="") as file:
            reader = csv.reader(file)
            self.column_names = next(reader)  # Read header
            self.data = [row for row in reader]  # Read data
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """

        with open(filename, mode="w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """

        seen = {}
        duplicates = []

        for i, row in enumerate(self.data):
            key = tuple(row[self.column_names.index(col)] for col in key_column_names)
            if key in seen:
                duplicates.append(i)
            else:
                seen[key] = i

        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        self.data = [row for row in self.data if "NA" not in row or None not in row]

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        column_values = [row[col_index] for row in self.data if row[col_index] != "NA"]

        if not column_values:
            return

        column_average = sum(column_values) / len(column_values)

        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = column_average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """

        summary_data = []

        for col_name in col_names:
            col_values = self.get_column(col_name, include_missing_values=False)
            col_values.sort()

            if not col_values:
                continue

            min_val = col_values[0]
            max_val = col_values[-1]
            mid_val = (min_val + max_val) / 2
            avg_val = sum(col_values) / len(col_values)
            median_val = (
                col_values[len(col_values) // 2]
                if len(col_values) % 2 != 0
                else (
                    col_values[len(col_values) // 2 - 1]
                    + col_values[len(col_values) // 2]
                )
                / 2
            )

            summary_data.append(
                [col_name, min_val, max_val, mid_val, avg_val, median_val]
            )

        return MyPyTable(
            column_names=["attribute", "min", "max", "mid", "avg", "median"],
            data=summary_data,
        )

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # Create a new table for the result
        result_table = MyPyTable()

        # Determine the columns for the result table
        result_table.column_names = list(self.column_names)
        for col in other_table.column_names:
            if col not in result_table.column_names:
                result_table.column_names.append(col)

        # Create a dictionary for quick lookup of rows in the other table
        other_table_dict = {}
        for row in other_table.data:
            key = tuple(
                row[other_table.column_names.index(col)] for col in key_column_names
            )
            if key in other_table_dict:
                other_table_dict[key].append(row)
            else:
                other_table_dict[key] = [row]

        # Perform the inner join
        for row in self.data:
            key = tuple(row[self.column_names.index(col)] for col in key_column_names)

            if key in other_table_dict:
                for other_row in other_table_dict[key]:
                    new_row = list(row)
                    for col in other_table.column_names:
                        if col not in self.column_names:
                            new_row.append(
                                other_row[other_table.column_names.index(col)]
                            )
                    result_table.data.append(new_row)

        return result_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        # Create a new table for the result
        result_table = MyPyTable()

        # Determine the columns for the result table
        result_table.column_names = list(self.column_names)
        for col in other_table.column_names:
            if col not in result_table.column_names:
                result_table.column_names.append(col)

        # Create dictionaries for quick lookup of rows in both tables
        self_table_dict = {}
        for row in self.data:
            key = tuple(row[self.column_names.index(col)] for col in key_column_names)
            if key in self_table_dict:
                self_table_dict[key].append(row)
            else:
                self_table_dict[key] = [row]

        other_table_dict = {}
        for row in other_table.data:
            key = tuple(
                row[other_table.column_names.index(col)] for col in key_column_names
            )
            if key in other_table_dict:
                other_table_dict[key].append(row)
            else:
                other_table_dict[key] = [row]

        # Debug: Print the dictionaries
        print("self_table_dict:", self_table_dict)
        print("other_table_dict:", other_table_dict)

        # Perform the full outer join
        all_keys = set(self_table_dict.keys()).union(set(other_table_dict.keys()))
        for key in all_keys:
            if key in self_table_dict and key in other_table_dict:
                for self_row in self_table_dict[key]:
                    for other_row in other_table_dict[key]:
                        new_row = list(self_row)
                        for col in other_table.column_names:
                            if col not in self.column_names:
                                new_row.append(
                                    other_row[other_table.column_names.index(col)]
                                )
                        result_table.data.append(new_row)
            elif key in self_table_dict:
                for self_row in self_table_dict[key]:
                    new_row = list(self_row)
                    for col in other_table.column_names:
                        if col not in self.column_names:
                            new_row.append("NA")
                    result_table.data.append(new_row)
            elif key in other_table_dict:
                for other_row in other_table_dict[key]:
                    new_row = ["NA"] * len(self.column_names)
                    for col in other_table.column_names:
                        if col in self.column_names:
                            new_row[self.column_names.index(col)] = other_row[
                                other_table.column_names.index(col)
                            ]
                        else:
                            new_row.append(
                                other_row[other_table.column_names.index(col)]
                            )
                    result_table.data.append(new_row)

        return result_table
