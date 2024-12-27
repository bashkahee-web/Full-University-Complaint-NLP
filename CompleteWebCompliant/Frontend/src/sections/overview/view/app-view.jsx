// import { faker } from '@faker-js/faker';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Unstable_Grid2';
import Typography from '@mui/material/Typography';

// import { LineChart } from '@mui/x-charts/LineChart';

import useComplaints from 'src/sections/finance/useComplaints';

import AppChartThree from '../AppChartThree';
import AppBarCompliants from '../AppBarCompliants';
import AppWidgetSummary from '../app-widget-summary';
import PieChartCompliants from '../PieChartCompliants';
// import SimpleLineChart from '../app-current-subject';
// import GaugeChart from '../app-current-subject';

// ----------------------------------------------------------------------

export default function AppView() {
  const { complaints: academicComplaints, isLoading: isLoadingAcademic, error: academicError } = useComplaints('academic');
  const { complaints: financeComplaints, isLoading: isLoadingFinance, error: financeError } = useComplaints('finance');
  const { complaints: equipmentComplaints, isLoading: isLoadingEquipment, error: equipmentError } = useComplaints('equipment');

  if (isLoadingAcademic || isLoadingFinance || isLoadingEquipment) {
      return <p>Loading complaints data...</p>;
  }

  if (academicError || financeError || equipmentError) {
      return <p>Error loading data. Please try again.</p>;
  }
  return (
    <Container maxWidth="xl">
      <Typography variant="h4" sx={{ mb: 5 }}>
        Hi, View Dashboard👋
      </Typography>

      <Grid container spacing={3}>
        <Grid xs={12} sm={6} md={4}>
          <AppWidgetSummary
            title="Academic Compliants"
            total={academicComplaints.length}
            color="success"
            icon={<img alt="icon" src="/assets/icons/glass/academic.png" />}
          />
        </Grid>

        <Grid xs={12} sm={6} md={4}>
          <AppWidgetSummary
            title="Finance Compliants"
            total={financeComplaints.length}
            color="info"
            icon={<img alt="icon" src="/assets/icons/glass/finance.png" />}
          />
        </Grid>

        <Grid xs={12} sm={6} md={4}>
          <AppWidgetSummary
            title="Equipment Compliants"
            total={equipmentComplaints.length}
            color="warning"
            icon={<img alt="icon" src="/assets/icons/glass/equipment.png" />}
          />
        </Grid>

     

        <Grid xs={12} md={6} lg={8}>
        <AppBarCompliants
    title="Number of Compliants by Category"
    subheader="Overview number of related to each complaints category"
    chart={{
        labels: ['Academic', 'Finance', 'Equipment'],
        series: [
            {
                name: 'Compliants',
                type: 'column',  // Assuming you still want columns, adjust as necessary
                fill: 'solid',
                data: [academicComplaints.length, financeComplaints.length, equipmentComplaints.length],  // Example data, replace with your actual data
            }
        ]
    }}
/>

        </Grid>

        <Grid xs={12} md={6} lg={4}>
          <PieChartCompliants
            title="Compliants as Categories"
            chart={{ series: [
              { label: 'Academic', value: academicComplaints ? academicComplaints.length : 0 },
              { label: 'Finance', value: financeComplaints ? financeComplaints.length : 0 },
              { label: 'Equipment', value: equipmentComplaints ? equipmentComplaints.length : 0 },
          ], }}
          />
        </Grid>

        <Grid xs={12} md={6} lg={8}>
          <AppChartThree
            title="Compliants"
            subheader="Chart for Compliants"
            chart={{
              series: [
                { label: 'Acadmeic', value: academicComplaints.length },
                { label: 'Finance', value: financeComplaints.length },
                { label: 'Equipment', value: equipmentComplaints.length },
              ],
            }}
          />
        </Grid>

        {/* <Grid xs={12} md={6} lg={4}>
          <GaugeChart/>
        </Grid> */}

      </Grid>
    </Container>
  );
}
