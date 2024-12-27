import PropTypes from 'prop-types';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardHeader from '@mui/material/CardHeader';

import Chart, { useChart } from 'src/components/chart';

export default function AppBarCompliants({ title, subheader, chart, ...other }) {
  const { labels, colors, series, options } = chart;

  // Update chart options for categorical data
  const chartOptions = useChart({
    colors,
    plotOptions: {
      bar: {
        columnWidth: '16%', // Adjust as needed for visual clarity
      },
    },
    fill: {
      type: series.map((i) => i.fill),
    },
    labels,
    xaxis: {
      type: 'category', // Changed from 'datetime' to 'category'
    },
    tooltip: {
      shared: true,
      intersect: false,
      y: {
        formatter: (value) => {
          if (typeof value !== 'undefined') {
            return `${value.toFixed(0)} Compliant`; // Ensures tooltip shows integer values
          }
          return value;
        },
      },
    },
    ...options,
  });

  return (
    <Card {...other}>
      <CardHeader title={title} subheader={subheader} />
      <Box sx={{ p: 3, pb: 1 }}>
        <Chart
          dir="ltr"
          type="line" // Consider changing if a different type is more appropriate for categories
          series={series}
          options={chartOptions}
          width="100%"
          height={364}
        />
      </Box>
    </Card>
  );
}

AppBarCompliants.propTypes = {
  chart: PropTypes.shape({
    labels: PropTypes.array.isRequired,
    colors: PropTypes.array,
    series: PropTypes.array.isRequired,
    options: PropTypes.object,
  }),
  subheader: PropTypes.string,
  title: PropTypes.string,
};
