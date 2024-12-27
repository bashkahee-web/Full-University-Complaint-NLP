import { Helmet } from 'react-helmet-async';

import { SignUpView } from 'src/sections/SignUp';



// ----------------------------------------------------------------------

export default function SignUpPage() {
  return (
    <>
      <Helmet>
        <title> SignUp </title>
      </Helmet>

      <SignUpView />
    </>
  );
}
