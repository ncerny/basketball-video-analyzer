/**
 * Navigation Component
 *
 * Navigation bar using Mantine components
 */

import { Group, Button, Box, Container } from '@mantine/core';
import { Link, useLocation } from 'react-router-dom';
import { IconVideo, IconUsers } from '@tabler/icons-react';

export function Navigation() {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <Box component="nav" mb="xl" style={{ borderBottom: '1px solid var(--mantine-color-dark-4)' }}>
      <Container size="xl" py="md">
        <Group gap="md">
          <Button
            component={Link}
            to="/"
            variant={isActive('/') ? 'filled' : 'subtle'}
            leftSection={<IconVideo size={18} />}
          >
            Games
          </Button>
          <Button
            component={Link}
            to="/players"
            variant={isActive('/players') ? 'filled' : 'subtle'}
            leftSection={<IconUsers size={18} />}
          >
            Players
          </Button>
        </Group>
      </Container>
    </Box>
  );
}
