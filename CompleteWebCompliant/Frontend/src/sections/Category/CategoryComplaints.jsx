// // In a React component
// import React from 'react';
// import PropTypes from 'prop-types';

// import useComplaints from '../finance/useComplaints';

// function CategoryComplaints({ category }) {
//     const { complaints, isLoading, error } = useComplaints(category);

//     if (isLoading) return <p>Loading...</p>;
//     if (error) return <p>Error: {error}</p>;

//     return (
//         <div>
//             <h1>Complaints in {category}</h1>
//             {complaints.length ? (
//                 <ul>
//                     {complaints.map(complaint => (
//                         <li key={complaint._id}>
//                             {complaint.description}
//                         </li>
//                     ))}
//                 </ul>
//             ) : <p>No complaints found.</p>}
//         </div>
//     );
// }

// CategoryComplaints.propTypes = {
//     category: PropTypes.string.isRequired
// };

// export default CategoryComplaints;
