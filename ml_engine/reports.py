import os
import json

from datetime import datetime

from jinja2 import (
    Environment,
    FileSystemLoader,
    select_autoescape
)

from auto_ml.ml_engine.config import (
    REPORT_DIR,
    REPORT_NAME
)


def create_environment():

    template_dir = os.path.join(

        os.path.dirname(__file__),

        "templates"

    )

    return Environment(

        loader=FileSystemLoader(

            template_dir

        ),

        autoescape=select_autoescape()

    )


def default_template():

    return """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>AutoML Report</title>

<style>

body{

font-family:Arial;

margin:40px;

background:#f7f7f7;

}

.section{

background:white;

padding:20px;

margin-bottom:20px;

border-radius:10px;

box-shadow:0 2px 5px rgba(0,0,0,.1);

}

table{

border-collapse:collapse;

width:100%;

}

td,th{

border:1px solid #ddd;

padding:8px;

}

th{

background:#eeeeee;

}

img{

width:700px;

margin-top:20px;

margin-bottom:20px;

border:1px solid #ddd;

}

h1{

color:#1f2937;

}

h2{

color:#2563eb;

}

</style>

</head>

<body>

<h1>AutoML Report</h1>

<p><b>Generated :</b> {{ generated }}</p>

<div class="section">

<h2>Dataset</h2>

<table>

<tr>

<th>Rows</th>

<th>Columns</th>

<th>Target</th>

<th>Problem Type</th>

</tr>

<tr>

<td>{{ dataset.rows }}</td>

<td>{{ dataset.columns }}</td>

<td>{{ dataset.target }}</td>

<td>{{ dataset.problem_type }}</td>

</tr>

</table>

</div>

<div class="section">

<h2>Best Model</h2>

<table>

{% for key,value in best_model.items() %}

<tr>

<td>{{ key }}</td>

<td>{{ value }}</td>

</tr>

{% endfor %}

</table>

</div>

<div class="section">

<h2>All Models</h2>

<table>

<tr>

{% for col in all_models[0].keys() %}

<th>{{ col }}</th>

{% endfor %}

</tr>

{% for row in all_models %}

<tr>

{% for value in row.values() %}

<td>{{ value }}</td>

{% endfor %}

</tr>

{% endfor %}

</table>

</div>

<div class="section">

<h2>Visualizations</h2>

{% for image in images %}

<h3>{{ image.title }}</h3>

<img src="{{ image.path }}">

{% endfor %}

</div>

</body>

</html>
"""



def generate_html_report(

    dataset,

    best_model,

    all_models,

    artifacts,

    output_directory=REPORT_DIR

):

    os.makedirs(

        output_directory,

        exist_ok=True

    )

    template = default_template()

    env = Environment()

    html = env.from_string(

        template

    ).render(

        generated=datetime.now().strftime(

            "%Y-%m-%d %H:%M:%S"

        ),

        dataset=dataset,

        best_model=best_model,

        all_models=all_models,

        images=[

            {

                "title": key.replace(

                    "_",

                    " "

                ).title(),

                "path": value

            }

            for key, value in artifacts.items()

            if value is not None

        ]

    )

    report_path = os.path.join(

        output_directory,

        REPORT_NAME

    )

    with open(

        report_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(

            html

        )

    return report_path


def save_metrics_json(

    metrics,

    output_directory=REPORT_DIR

):

    os.makedirs(

        output_directory,

        exist_ok=True

    )

    output = os.path.join(

        output_directory,

        "metrics.json"

    )

    with open(

        output,

        "w"

    ) as file:

        json.dump(

            metrics,

            file,

            indent=4

        )

    return output


def save_summary_json(

    summary,

    output_directory=REPORT_DIR

):

    os.makedirs(

        output_directory,

        exist_ok=True

    )

    output = os.path.join(

        output_directory,

        "summary.json"

    )

    with open(

        output,

        "w"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4

        )

    return output


def build_report_data(

    dataset_shape,

    target_column,

    problem_type

):

    return {

        "rows": dataset_shape[0],

        "columns": dataset_shape[1],

        "target": target_column,

        "problem_type": problem_type

    }


def generate_report(

    dataset_shape,

    target_column,

    problem_type,

    best_model,

    all_models,

    artifacts

):

    dataset = build_report_data(

        dataset_shape,

        target_column,

        problem_type

    )

    report = generate_html_report(

        dataset,

        best_model,

        all_models,

        artifacts

    )

    metrics = save_metrics_json(

        {

            "best_model": best_model,

            "all_models": all_models

        }

    )

    summary = save_summary_json(

        dataset

    )

    return {

        "report": report,

        "metrics": metrics,

        "summary": summary

    }