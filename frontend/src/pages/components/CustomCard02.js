import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { Button, CardActionArea, CardActions } from '@mui/material';

export default function DefaultCard02() {
  return (
    <Card sx={{ maxWidth: "100%", height: "350px" }}>
      <CardActionArea>
        <CardMedia
          component="img"
          height="100"
          image="/static images/eye03.webp"
          alt="green iguana"
        />
        <CardContent>
          <Typography gutterBottom variant="h5" component="div">
          Retina Diseases
                    </Typography>
          <Typography variant="body2" color="text.secondary">
          Retina diseases, such as diabetic retinopathy and macular degeneration, can lead to vision impairment. Early diagnosis and treatments like medications, lasers, or surgery can help protect vision.           </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}