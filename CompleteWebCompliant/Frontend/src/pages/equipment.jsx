import { Helmet } from 'react-helmet-async';

import { EquipmentView } from 'src/sections/equipment';





// ----------------------------------------------------------------------

export default function AppPage() {
  return (
    <>
      <Helmet>
        <title> Equipment | Minimal UI </title>
      </Helmet>

      <EquipmentView />
    </>
  );
}
