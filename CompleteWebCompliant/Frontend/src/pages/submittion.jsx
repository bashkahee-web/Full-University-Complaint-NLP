import { Helmet } from 'react-helmet-async';

import { SubmitView } from 'src/sections/submittion';



// ----------------------------------------------------------------------

export default function ProductsPage() {
  return (
    <>
      <Helmet>
        <title> Submittion | Minimal UI </title>
      </Helmet>

      <SubmitView/>
    </>
  );
}
