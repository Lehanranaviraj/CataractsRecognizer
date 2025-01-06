import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { Button, CardActionArea, CardActions } from '@mui/material';

export default function DefaultCard03() {
  return (
    <Card sx={{ maxWidth: "100%", height: "350px" }}>
      <CardActionArea>
        <CardMedia
          component="img"
          height="100"
          image="/static images/eye04.jpg"
          alt="green iguana"
        />
        <CardContent>
          <Typography gutterBottom variant="h5" component="div">
          Maintaining Healthy Eyes          </Typography>
          <Typography variant="body2" color="text.secondary">
          Adopt healthy habits like a nutritious diet, regular eye exams, and reducing screen time. Protect your eyes to prevent diseases and ensure clear vision throughout life.
</Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}