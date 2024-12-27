import PropTypes from 'prop-types';
import React, { useState, useEffect, createContext } from 'react';

import { fetchRealUsers } from './user';

export const UserContext = createContext();

export const UserProvider = ({ children }) => {
    const [currentUser, setCurrentUser] = useState({
        displayName: 'Loading...',
        photoURL: '/assets/images/avatars/avatar_0.jpg'
    });

    useEffect(() => {
        const loadUsers = async () => {
            const users = await fetchRealUsers();
            if (users.length > 0) {
                // Assuming the last user in array is the latest to sign in
                const latestUser = users[users.length - 1];
                const newUser = {
                    displayName: latestUser.name,
                    photoURL: latestUser.avatarUrl
                };
                setCurrentUser(newUser);
                // Update localStorage with the latest user data every time it changes
                localStorage.setItem('user', JSON.stringify(newUser));
            }
        };

        loadUsers();
    }, []);

    return (
        <UserContext.Provider value={currentUser}>
            {children}
        </UserContext.Provider>
    );
};

UserProvider.propTypes = {
    children: PropTypes.node.isRequired // Define the prop types for 'children'
};
