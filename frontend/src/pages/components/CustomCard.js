import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { Button, CardActionArea, CardActions } from '@mui/material';

export default function DefaultCard() {
  return (
    <Card sx={{ maxWidth: "100%", height: "350px" }}>
      <CardActionArea>
        <CardMedia
          component="img"
          height="100"
          image="/static images/eye01.png"
          alt="eye image"
        />
        <CardContent>
          <Typography gutterBottom variant="h5" component="div">
          Cataracts
          </Typography>
          <Typography variant="body2" color="text.secondary">
          Cataracts cause cloudiness in the eye’s lens, leading to blurry vision. Understand causes, symptoms, and treatments like prescription lenses and surgical options to restore clear vision.          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}