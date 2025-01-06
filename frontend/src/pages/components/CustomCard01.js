import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { Button, CardActionArea, CardActions } from '@mui/material';

export default function DefaultCard01() {
  return (
    <Card sx={{ maxWidth: "100%", height: "350px" }}>
      <CardActionArea>
        <CardMedia
          component="img"
          height="100"
          image="/static images/eye02.jpg"
          alt="green iguana"
        />
        <CardContent>
          <Typography gutterBottom variant="h5" component="div">
          Glaucoma 
          </Typography>
          <Typography variant="body2" color="text.secondary"> 
          Glaucoma damages the optic nerve due to high eye pressure. Learn about its risk factors and symptoms, and explore treatments like medications, laser therapy, and surgeries to prevent vision loss.            </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}