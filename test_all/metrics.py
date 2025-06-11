#from anls import anls_score
from scipy import optimize
from typing import Optional
import pandas as pd
import dataclasses
import numpy as np
import editdistance
import itertools

from datetime import datetime
from pytz import timezone

# RMS
def _to_float(text):  # 단위 떼고 숫자만..?
  try:
    if text.endswith("%"):
      # Convert percentages to floats.
      return float(text.rstrip("%")) / 100.0
    else:
      return float(text)
  except ValueError:
    return None


def _get_relative_distance(
    target, prediction, theta = 1.0
):
  """Returns min(1, |target-prediction|/|target|)."""
  if not target:
    return int(not prediction)
  distance = min(abs((target - prediction) / target), 1)
  return distance if distance < theta else 1

def anls_metric(target: str, prediction: str, theta: float = 0.5):
    edit_distance = editdistance.eval(target, prediction)
    normalize_ld = edit_distance / max(len(target), len(prediction))
    return 1 - normalize_ld if normalize_ld < theta else 0

def _permute(values, indexes):
    return tuple(values[i] if i < len(values) else "" for i in indexes)


@dataclasses.dataclass(frozen=True)
class Table:
  """Helper class for the content of a markdown table."""

  base: Optional[str] = None
  title: Optional[str] = None
  chartType: Optional[str] = None
  headers: tuple[str, Ellipsis] = dataclasses.field(default_factory=tuple)
  rows: tuple[tuple[str, Ellipsis], Ellipsis] = dataclasses.field(default_factory=tuple)

  def permuted(self, indexes):
    """Builds a version of the table changing the column order."""
    return Table(
        base=self.base,
        title=self.title,
        chartType=self.chartType,
        headers=_permute(self.headers, indexes),
        rows=tuple(_permute(row, indexes) for row in self.rows),
    )

  def aligned(
      self, headers, text_theta = 0.5
  ):
    """Builds a column permutation with headers in the most correct order."""
    if len(headers) != len(self.headers):
      raise ValueError(f"Header length {headers} must match {self.headers}.")
    distance = []
    for h2 in self.headers:
      distance.append(
          [
              1 - anls_metric(h1, h2, text_theta)
              for h1 in headers
          ]
      )
    cost_matrix = np.array(distance)
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
    permutation = [idx for _, idx in sorted(zip(col_ind, row_ind))]
    score = (1 - cost_matrix)[permutation[1:], range(1, len(row_ind))].prod()
    return self.permuted(permutation), score

def _parse_table(text, transposed = False): # 표 제목, 열 이름, 행 찾기
  """Builds a table from a markdown representation."""
  lines = text.lower().splitlines()
  if not lines:
    return Table()
  
  if lines[0].startswith("대상: "):
      base = lines[0][len("대상: ") :].strip()
      offset = 1 #
  else:
    base = None
    offset = 0
  if lines[1].startswith("제목: "):
    title = lines[1][len("제목: ") :].strip()
    offset = 2 #
  else:
    title = None
    offset = 1
  if lines[2].startswith("유형: "):
    chartType = lines[2][len("유형: ") :].strip()
    offset = 3 #
  else:
    chartType = None
  
  if len(lines) < offset + 1:
    return Table(base=base, title=title, chartType=chartType)
    
  rows = []
  for line in lines[offset:]:
    rows.append(tuple(v.strip() for v in line.split(" | ")))
  if transposed:
    rows = [tuple(row) for row in itertools.zip_longest(*rows, fillvalue="")]
  return Table(base=base, title=title, chartType=chartType, headers=rows[0], rows=tuple(rows[1:]))

def _get_table_datapoints(table):
    datapoints = {}
    if table.base is not None:
        datapoints["대상"] = table.base
    if table.title is not None:
      datapoints["제목"] = table.title
    if table.chartType is not None:
      datapoints["유형"] = table.chartType
    if not table.rows or len(table.headers) <= 1:
        return datapoints
    for row in table.rows:
        for header, cell in zip(table.headers[1:], row[1:]):
            #print(f"{row[0]} {header} >> {cell}")
            datapoints[f"{row[0]} {header}"] = cell #
    return datapoints

def _get_datapoint_metric(  # 
    target,
    prediction,
    text_theta=0.5,
    number_theta=0.1,
):
  """Computes a metric that scores how similar two datapoint pairs are."""
  key_metric = anls_metric(
      target[0], prediction[0], text_theta
  )
  pred_float = _to_float(prediction[1]) # 숫자인지 확인
  target_float = _to_float(target[1])
  if pred_float is not None and target_float:
    return key_metric * (
        1 - _get_relative_distance(target_float, pred_float, number_theta)  # 숫자면 상대적 거리값 계산
    )
  elif target[1] == prediction[1]:
    return key_metric
  else:
    return key_metric * anls_metric(
        target[1], prediction[1], text_theta
    )

def _table_datapoints_precision_recall_f1(  # 찐 계산
    target_table,
    prediction_table,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """Calculates matching similarity between two tables as dicts."""
  target_datapoints = list(_get_table_datapoints(target_table).items())
  prediction_datapoints = list(_get_table_datapoints(prediction_table).items())
  if not target_datapoints and not prediction_datapoints:
    return 1, 1, 1
  if not target_datapoints:
    return 0, 1, 0
  if not prediction_datapoints:
    return 1, 0, 0
  distance = []
  for t, _ in target_datapoints:
    distance.append(
        [
            1 - anls_metric(t, p, text_theta)
            for p, _ in prediction_datapoints
        ]
    )
  cost_matrix = np.array(distance)
  row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
  score = 0
  for r, c in zip(row_ind, col_ind):
    score += _get_datapoint_metric(
        target_datapoints[r], prediction_datapoints[c], text_theta, number_theta
    )
  if score == 0:
    return 0, 0, 0
  precision = score / len(prediction_datapoints)
  recall = score / len(target_datapoints)
  return precision, recall, 2 * precision * recall / (precision + recall)

def table_datapoints_precision_recall_per_point(  # 각각 계산...
    targets,
    predictions,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """Computes precisin recall and F1 metrics given two flattened tables.

  Parses each string into a dictionary of keys and values using row and column
  headers. Then we match keys between the two dicts as long as their relative
  levenshtein distance is below a threshold. Values are also compared with
  ANLS if strings or relative distance if they are numeric.

  Args:
    targets: list of list of strings.
    predictions: list of strings.
    text_theta: relative edit distance above this is set to the maximum of 1.
    number_theta: relative error rate above this is set to the maximum of 1.

  Returns:
    Dictionary with per-point precision, recall and F1
  """
  assert len(targets) == len(predictions)
  per_point_scores = {"precision": [], "recall": [], "f1": []}
  for pred, target in zip(predictions, targets):
    all_metrics = []
    for transposed in [True, False]:
      pred_table = _parse_table(pred, transposed=transposed)
      target_table = _parse_table(target, transposed=transposed)

      all_metrics.extend([_table_datapoints_precision_recall_f1(target_table, pred_table, text_theta, number_theta)])
      
    p, r, f = max(all_metrics, key=lambda x: x[-1])
    per_point_scores["precision"].append(p)
    per_point_scores["recall"].append(r)
    per_point_scores["f1"].append(f)
  return per_point_scores

def table_datapoints_precision_recall(  # deplot 성능지표
    targets,
    predictions,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """Aggregated version of table_datapoints_precision_recall_per_point().

  Same as table_datapoints_precision_recall_per_point() but returning aggregated
  scores instead of per-point scores.

  Args:
    targets: list of list of strings.
    predictions: list of strings.
    text_theta: relative edit distance above this is set to the maximum of 1.
    number_theta: relative error rate above this is set to the maximum of 1.

  Returns:
    Dictionary with aggregated precision, recall and F1
  """
  score_dict = table_datapoints_precision_recall_per_point(
      targets, predictions, text_theta, number_theta
  )
  return {
      "table_datapoints_precision": (
          100.0 * sum(score_dict["precision"]) / len(targets)
      ),
      "table_datapoints_recall": (
          100.0 * sum(score_dict["recall"]) / len(targets)
      ),
      "table_datapoints_f1": 100.0 * sum(score_dict["f1"]) / len(targets),
  }
  
  
# RNSS
def _get_table_numbers(text): # 데이터 값(숫자) 추출 
  numbers = []
  for line in text.splitlines():  # 줄 단위 split
    for part in line.split(" | "):
      if part.strip():
        try:
          numbers.append(float(part))
        except ValueError:
          pass
  return numbers

def _table_numbers_match(target, prediction):
  """Calculates matching similarity between two tables following ChartQA."""

  target_numbers = _get_table_numbers(target)
  prediction_numbers = _get_table_numbers(prediction)
  if not target_numbers and not prediction_numbers:
    return 1
  if not target_numbers or not prediction_numbers:
    return 0
  max_len = max(len(target_numbers), len(prediction_numbers))
  distance = []
  for t in target_numbers:
    distance.append([_get_relative_distance(t, p) for p in prediction_numbers])
  cost_matrix = np.array(distance)
  row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
  return 1 - cost_matrix[row_ind, col_ind].sum() / max_len

def table_number_accuracy_per_point(
    targets,
    predictions,
):
  """Calculates matching similarity between two tables following ChartQA.

  Keeps only numbers and performas a linear matching using the relative error.

  Args:
    targets: ground truth text.
    predictions: predicted text.

  Returns:
    A list of float numbers.
  """
  all_points_scores = []
  for p, targets in zip(predictions, targets):
    all_points_scores.append(max(_table_numbers_match(t, p) for t in targets))
  return all_points_scores

def table_number_accuracy(
    targets,
    predictions,
):
  """Aggregated version of table_number_accuracy_per_point().

  Same as table_number_accuracy_per_point() but returning an aggregated score.

  Args:
    targets: ground truth text.
    predictions: predicted text.

  Returns:
    dictionary with metric names as keys and metric value as values.
  """
  scores = table_number_accuracy_per_point(targets, predictions)
  return {"numbers_match": (100.0 * sum(scores)) / len(targets)}