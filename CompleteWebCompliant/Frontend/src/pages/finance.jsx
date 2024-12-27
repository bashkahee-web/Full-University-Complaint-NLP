import { Helmet } from 'react-helmet-async';

import {FinanceView} from 'src/sections/finance';





// ----------------------------------------------------------------------

export default function AppPage() {
  return (
    <>
      <Helmet>
        <title> Finance | Minimal UI </title>
      </Helmet>

      <FinanceView />
    </>
  );
}
