function createChart(ctx, n_epochs) {
  chart = Chart.Line(ctx, {
    data: {
      labels: Array.from(Array(parseInt(n_epochs)), (_, i) => i + 1),
      datasets: [{
        label: 'Train Loss',
        borderColor: window.chartColors.red,
        backgroundColor: window.chartColors.red,
        fill: false,
        data: [],
        yAxisID: 'y-axis-1',
        radius: 1
      }, {
        label: 'Train Acc',
        borderColor: window.chartColors.blue,
        backgroundColor: window.chartColors.blue,
        fill: false,
        data: [],
        yAxisID: 'y-axis-2',
        radius: 1
      }, {
        label: 'Val Loss',
        borderColor: window.chartColors.green,
        backgroundColor: window.chartColors.green,
        fill: false,
        data: [],
        yAxisID: 'y-axis-1',
        radius: 1
      }, {
        label: 'Val Acc',
        borderColor: window.chartColors.yellow,
        backgroundColor: window.chartColors.yellow,
        fill: false,
        data: [],
        yAxisID: 'y-axis-2',
        radius: 1
      }]
    },
    options: {
      responsive: true,
      hoverMode: 'index',
      stacked: false,
      title: {
        display: false,
        text: 'Training'
      },
      /*
      elements: {
        point: {
          radius: [1, 4, 1, 1, 1], //customRadius,
          display: true
        },
      },
      */
      scales: {
        yAxes: [{
          type: 'linear', // only linear but allow scale type registration. This allows extensions to exist solely for log scale for instance
          display: true,
          position: 'left',
          id: 'y-axis-1',
          ticks: {
            suggestedMin: 0,
            // suggestedMax: 1
          }
        }, {
          type: 'linear', // only linear but allow scale type registration. This allows extensions to exist solely for log scale for instance
          display: true,
          position: 'right',
          id: 'y-axis-2',
          ticks: {
            suggestedMin: 0,
            suggestedMax: 1,
          },

          // grid line settings
          gridLines: {
            drawOnChartArea: false, // only want the grid lines for one axis to show up
          },
        }],
      }
    }
  });

  return chart;
}

function addChartData(chart, train_acc, train_loss, val_acc, val_loss) {
  chart.data.datasets[0].data.push(train_loss);
  chart.data.datasets[1].data.push(train_acc);
  chart.data.datasets[2].data.push(val_loss);
  chart.data.datasets[3].data.push(val_acc);
}

function createRadiusList(length, epoch, size) {
  radius = Array(length).fill(1);
  radius[epoch-1] = size;

  return radius;
}

function setChartData(chart, data, best_train_epoch, best_val_epoch) {
  chart.data.datasets[0].data = data[0];
  chart.data.datasets[1].data = data[1];
  chart.data.datasets[2].data = data[2];
  chart.data.datasets[3].data = data[3];

  let train_radius_list = createRadiusList(data[0].length, best_train_epoch, 4);
  let val_radius_list = createRadiusList(data[2].length, best_val_epoch, 4)
  chart.data.datasets[0].radius = train_radius_list;
  chart.data.datasets[1].radius = train_radius_list;
  chart.data.datasets[2].radius = val_radius_list;
  chart.data.datasets[3].radius = val_radius_list;
}